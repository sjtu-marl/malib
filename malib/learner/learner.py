# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, Any, Tuple, Callable, Type, List, Union
from abc import ABC, abstractmethod
from collections import deque

import traceback

import gym
import torch
import ray

from ray.util.queue import Queue
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.utils.tianshou_batch import Batch
from malib.utils.monitor import write_to_tensorboard
from malib.remote.interface import RemoteInterface
from malib.rl.common.trainer import Trainer
from malib.common.task import OptimizationTask
from malib.common.strategy_spec import StrategySpec
from malib.backend.dataset_server.data_loader import DynamicDataset


class Learner(RemoteInterface, ABC):
    """Base class of agent interface, for training"""

    @abstractmethod
    def __init__(
        self,
        experiment_tag: str,
        runtime_id: str,
        log_dir: str,
        observation_space: gym.Space,
        action_space: gym.Space,
        algorithms: Dict[str, Tuple[Type, Type, Dict, Dict]],
        agent_mapping_func: Callable[[AgentID], str],
        governed_agents: Tuple[AgentID],
        trainer_config: Dict[str, Any],
        custom_config: Dict[str, Any] = None,
        local_buffer_config: Dict = None,
        verbose: bool = True,
        dataset: DynamicDataset = None,
    ):
        """Construct agent interface for training.

        Args:
            experiment_tag (str): Experiment tag.
            runtime_id (str): Assigned runtime id, should be an element of the agent mapping results.
            log_dir (str): The directory for logging.
            observation_space (gym.Space): Observation space.
            action_space (gym.Space): Action space.
            algorithms (Dict[str, Tuple[Type, Type, Dict]]): A dict that describes the algorithm candidates. Each is \
                a tuple of `policy_cls`, `trainer_cls`, `model_config` and `custom_config`.
            agent_mapping_func (Callable[[AgentID], str]): A function that defines the rule of agent groupping.
            governed_agents (Tuple[AgentID]): A tuple that records which agents is related to this learner. \
                Note that it should be a subset of the original set of environment agents.
            trainer_config (Dict[str, Any]): Trainer configuration.
            custom_config (Dict[str, Any], optional): A dict of custom configuration. Defaults to None.
            local_buffer_config (Dict, optional): A dict for local buffer configuration. Defaults to None.
            verbose (bool, True): Enable logging or not. Defaults to True.
        """

        if verbose:
            Logger.info("\tAssigned GPUs: {}".format(ray.get_gpu_ids()))

        local_buffer_config = local_buffer_config or {}
        device = torch.device("cuda" if ray.get_gpu_ids() else "cpu")

        # initialize a strategy spec for policy maintainance.
        strategy_spec = StrategySpec(
            policy_cls=algorithms["default"][0],
            observation_space=observation_space,
            action_space=action_space,
            model_config=algorithms["default"][2],
            **algorithms["default"][3],
        )

        self._runtime_id = runtime_id
        self._device = device
        self._algorithms = algorithms
        self._governed_agents = governed_agents
        self._strategy_spec = strategy_spec
        self._agent_mapping_func = agent_mapping_func
        self._custom_config = custom_config

        self._summary_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self._trainer_config = trainer_config
        self._total_step = 0
        self._total_epoch = 0
        self._trainer: Trainer = algorithms["default"][1](trainer_config)
        self._policies = {}

        dataset = dataset or self.create_dataset()
        self._data_loader = DataLoader(dataset, batch_size=trainer_config["batch_size"])
        self._active_tups = deque()

        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def data_loader(self) -> DataLoader:
        return self._data_loader

    @property
    def governed_agents(self) -> Tuple[str]:
        """Return a tuple of governed environment agents.

        Returns:
            Tuple[str]: A tuple of agent ids.
        """

        return tuple(self._governed_agents)

    @property
    def device(self) -> Union[str, torch.DeviceObjType]:
        """Retrive device name.

        Returns:
            Union[str, torch.DeviceObjType]: Device name.
        """

        return self._device

    def create_dataset(self) -> DynamicDataset:
        raise NotImplementedError

    def add_policies(self, n: int) -> StrategySpec:
        """Construct `n` new policies and return the latest strategy spec.

        Args:
            n (int): Indicates how many new policies will be added.

        Returns:
            StrategySpec: The latest strategy spec instance.
        """

        for _ in range(n):
            spec_pid = f"policy-{len(self._strategy_spec)}"
            self._strategy_spec.register_policy_id(policy_id=spec_pid)
            policy = self._strategy_spec.gen_policy()
            policy_id = f"{self._strategy_spec.id}/{spec_pid}"
            self._policies[policy_id] = policy
            # active tups store the policy info tuple for training, the
            # the data request relies on it.
            self._active_tups.append((self._strategy_spec.id, spec_pid))
            self._trainer.reset(policy_instance=policy)

            ray.get(self._parameter_server.create_table.remote(self._strategy_spec))
            ray.get(
                self._parameter_server.set_weights.remote(
                    spec_id=self._strategy_spec.id,
                    spec_policy_id=spec_pid,
                    state_dict=policy.state_dict(),
                )
            )

        return self._strategy_spec

    def push(self):
        """Push local weights to remote server"""

        pending_tasks = []
        for spec_pid in self._strategy_spec.policy_ids:
            pid = f"{self._strategy_spec.id}/{spec_pid}"
            task = self._parameter_server.set_weights.remote(
                spec_id=self._strategy_spec.id,
                spec_policy_id=spec_pid,
                state_dict=self._policies[pid].state_dict(),
            )
            pending_tasks.append(task)
        while len(pending_tasks) > 0:
            dones, pending_tasks = ray.wait(pending_tasks)

    def pull(self):
        """Pull remote weights to update local version."""

        pending_tasks = []

        for spec_pid in self._strategy_spec.policy_ids:
            pid = f"{self._strategy_spec.id}/{spec_pid}"
            task = self._parameter_server.get_weights.remote(
                spec_id=self._strategy_spec.id, spec_policy_id=spec_pid
            )
            pending_tasks.append(task)

        while len(pending_tasks) > 0:
            dones, pending_tasks = ray.wait(pending_tasks)
            for done in ray.get(dones):
                pid = "{}/{}".format(done["spec_id"], done["spec_policy_id"])
                self._policies[pid].load_state_dict(done["weights"])

    @abstractmethod
    def multiagent_post_process(
        self,
        batch_info: Union[
            Dict[AgentID, Tuple[Batch, List[int]]], Tuple[Batch, List[int]]
        ],
    ) -> Dict[str, Any]:
        """Merge agent buffer here and return the merged buffer.

        Args:
            batch_info (Union[Dict[AgentID, Tuple[Batch, List[int]]], Tuple[Batch, List[int]]]): Batch info, could be a dict of agent batch info or a tuple.

        Returns:
            Dict[str, Any]: A merged buffer dict.
        """

    def get_interface_state(self) -> Dict[str, Any]:
        """Return a dict that describes the current learning state.

        Returns:
            Dict[str, Any]: A dict of learning state.
        """

        return {
            "total_step": self._total_step,
            "total_epoch": self._total_epoch,
            "policy_num": len(self._strategy_spec),
            "active_tups": list(self._active_tups),
        }

    def sync_remote_parameters(self):
        """Push latest network parameters of active policies to remote parameter server."""

        top_active_tup = self._active_tups[0]
        ray.get(
            self._parameter_server.set_weights.remote(
                spec_id=top_active_tup[0],
                spec_policy_id=top_active_tup[1],
                state_dict=self._trainer.policy.state_dict(device="cpu"),
            )
        )

    def train(self, task: OptimizationTask) -> Dict[str, Any]:
        """Executes a optimization task and returns the final interface state.

        Args:
            stopping_conditions (Dict[str, Any]): Control the training stepping.
            reset_tate (bool, optional): Reset interface state or not. Default is True.

        Returns:
            Dict[str, Any]: A dict that describes the final state.
        """

        # XXX(ming): why we need to reset the state here? I think it is not necessary as
        #   an optimization task should be independent with other tasks.

        self.set_running(True)

        try:
            while self.is_running():
                for data in self.data_loader:
                    batch_info = self.multiagent_post_process(data)
                    step_info_list = self._trainer(batch_info)
                    for step_info in step_info_list:
                        self._total_step += 1
                        write_to_tensorboard(
                            self._summary_writer,
                            info=step_info,
                            global_step=self._total_step,
                            prefix=f"Training/{self._runtime_id}",
                        )
                    self.sync_remote_parameters()
                    self._total_epoch += 1
        except Exception as e:
            Logger.warning(
                f"training pipe is terminated. caused by: {traceback.format_exc()}"
            )

        if self.verbose:
            Logger.info(
                "training meets stopping condition after {} epoch(s), {} iteration(s)".format(
                    self._total_epoch, self._total_step
                )
            )
        return self.get_interface_state()

    def reset(self):
        """Reset training state."""

        self._total_step = 0
        self._total_epoch = 0
