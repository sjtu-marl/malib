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


from random import random
from typing import Dict, Any, Tuple, Callable, Type

import os
import time

from collections import deque

import torch
import ray
import reverb

from torch.utils import tensorboard

from malib import settings
from malib.utils.stopping_conditions import get_stopper
from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.remote.interface import RemoteInterFace
from malib.algorithm.common.trainer import Trainer
from malib.common.strategy_spec import StrategySpec
from malib.monitor.utils import write_to_tensorboard


class AgentInterface(RemoteInterFace):
    """Base class of agent interface, for training"""

    def __init__(
        self,
        experiment_tag: str,
        runtime_id: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Tuple[Dict, Dict, Dict]],
        agent_mapping_func: Callable[[AgentID], str],
        governed_agents: Tuple[AgentID],
        trainer_config: Dict[str, Any],
        custom_config: Dict[str, Any] = None,
        local_buffer_config: Dict = None,
    ):
        """Construct agent interface for training.

        Args:
            experiment_tag (str): Experiment tag.
            runtime_id (str): Assigned runtime id, should be an element of the agent mapping results.
            log_dir (str): The directory for logging.
            env_desc (Dict[str, Any]): A dict that describes the environment property.
            algorithms (Dict[str, Tuple[Dict, Dict, Dict]]): A dict that describes the algorithm candidates. Each is \
                a tuple which describes the policy config, and training config.
            agent_mapping_func (Callable[[AgentID], str]): A function that defines the rule of agent groupping.
            governed_agents (Tuple[AgentID]): A tuple that records which agents is related to this training procedures. \
                Note that it should be a subset of the original set of environment agents.
            trainer_config (Dict[str, Any]): Trainer configuration.
            custom_config (Dict[str, Any], optional): A dict of custom configuration. Defaults to None.
            local_buffer_config (Dict, optional): A dict for local buffer configuration. Defaults to None.
        """

        Logger.info("\tray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        Logger.info(
            "\tCUDA_VISIBLE_DEVICES: {}".format(
                os.environ.get("CUDA_VISIBLE_DEVICES", [])
            )
        )

        local_buffer_config = local_buffer_config or {}
        device = torch.device("cuda" if ray.get_gpu_ids() else "cpu")
        # a strategy spec dict, mapping from algorithm
        strategy_spec = StrategySpec(
            identifier=runtime_id,
            policy_ids=[],
            meta_data={
                "policy_cls": algorithms["default"],
                "experiment_tag": experiment_tag,
                "kwargs": {},
            },
        )

        self._runtime_id = runtime_id
        self._device = device
        self._env_desc = env_desc
        self._algorithms = algorithms
        self._governed_agents = governed_agents
        self._strategy_spec = strategy_spec
        self._agent_mapping_func = agent_mapping_func
        self._custom_config = custom_config

        self._summary_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self._trainer_config = trainer_config
        self._total_step = 0
        self._total_epoch = 0
        self._trainer: Trainer = None
        self._policies = {}

        self._offline_dataset = None
        self._coordinator = None
        self._parameter_server = None
        self._active_tups = deque()
        self._clients: Dict[str, Type[reverb.Client]] = {}

        # TODO(ming): local buffer is muted temporary, consider to recover it in future.

    def connect(self):
        """Connect backend server"""

        while True:
            try:
                if self._coordinator is None:
                    self._coordinator = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)

                if self._parameter_server is None:
                    self._parameter_server = ray.get_actor(
                        settings.PARAMETER_SERVER_ACTOR
                    )
            except Exception as e:
                Logger.debug(f"Waiting for coordinator server... {e}")
                time.sleep(1)
                continue

    def add_policies(self, n: int) -> StrategySpec:
        """Construct `n` new policies and return the latest strategy spec.

        Args:
            n (int): Indicates how many new policies will be added.

        Returns:
            StrategySpec: The latest strategy spec instance.
        """

        for _ in range(n):
            spec_pid = f"policy-{len(self._strategy_spec.policy_ids)}"
            self._strategy_spec.register_policy_id(policy_id=spec_pid)
            # TODO(ming): gen policy? from where?
            policy = self._strategy_spec.gen_policy()
            policy_id = f"{self._strategy_spec.id}/{spec_pid}"
            self._policies[policy_id] = policy
            self._active_tups.append((self._strategy_spec.id, spec_pid))

            ray.get(self._parameter_server.create_table.remote(self._strategy_spec))
            ray.get(
                self._parameter_server.set_weights.remote(
                    table_name=policy_id, state_dict=policy.state_dict()
                )
            )
            # TODO(ming): create trainer here?

        return self._strategy_spec

    def get_algorithm(self, key) -> Any:
        return self._algorithms[key]

    def get_algorthms(self) -> Dict[str, Any]:
        return self._algorithms

    def push(self):
        """Push local weights to remote server"""

        pending_tasks = []
        for spec_pid in self._strategy_spec.policy_ids:
            pid = f"{self._strategy_spec.id}/{spec_pid}"
            task = self._parameter_server.set_weights.remote(
                spec_id=self._strategy_spec.id,
                spec_policy_id=spec_pid,
                state_dict=self._policies[pid],
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
                spec_id=self._strategy_spec.id,
                spec_policy_id=spec_pid,
                state_dict=self._policies[pid],
            )
            pending_tasks.append(task)
        while len(pending_tasks) > 0:
            dones, pending_tasks = ray.wait(pending_tasks)
            for done in ray.get(dones):
                pid = "{}/{}".format(done["spec_id"], done["spec_policy_id"])
                self.policy[pid].load_state_dict(done["weights"])

    def request_data(self) -> Tuple[Any, str]:
        batch = {}
        for active_tup in list(self._active_tups):
            identifier = f"{active_tup[0]}/{active_tup[1]}"
            while identifier not in self._clients:
                # build client
                client_kwargs = ray.get(
                    self._offline_dataset.get_client_kwargs.remote(
                        table_name=identifier
                    )
                )
                if client_kwargs["address"] is not None:
                    self._clients[identifier] = reverb.Client(
                        server_address=client_kwargs["address"]
                    )
                else:
                    secs = random()
                    Logger.warning(
                        f"reverb server for {identifier} is not initialized yet, sleep {secs}(s) secionds"
                    )
                    time.sleep(secs)
            client = self._clients[identifier]
            batch[identifier] = client.sample(
                table=None,
                num_samples=1,
                emit_timesteps=True,
                unpack_as_table_signature=False,
            )
        batch = self.multiagent_post_process(batch)
        return batch

    def multiagent_post_process(
        self, batch: Dict[AgentID, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge agent buffer here.

        Args:
            batch (Dict[AgentID, Dict[str, Any]]): Agent buffer dict.

        Returns:
            Dict[str, Any]: A merged buffer dict.
        """

        return batch

    def get_interface_state(self):
        return {
            "total_step": self._total_step,
            "total_epoch": self._total_epoch,
            "policy_num": len(self._strategy_spec.policy_ids),
        }

    def train(self, reset_state: bool = True) -> Dict[str, Any]:
        """Executes training task and returns the final interface state.

        Args:
            reset_tate (bool, optional): Reset interface state or not. Default is True.

        Returns:
            Dict[str, Any]: A dict that describes the final state.
        """

        if reset_state:
            self.reset()

        stopper = get_stopper(conditions=self._stopping_conditions)

        # loop epoch
        while True:
            batch = self.request_data()
            step_info_list = self._trainer(batch)
            for step_info in step_info_list:
                self._total_step += 1
                write_to_tensorboard(
                    self._summary_writer,
                    info=step_info,
                    global_step=self._total_step,
                    prefix=f"Training/{self._runtime_id}",
                )
                if stopper.is_meet_condition(step_info):
                    self._total_epoch += 1
                    break

            if stopper.is_meet_condition(step_info):
                break
            self._total_epoch += 1

        return self.get_interface_state()

    def reset(self):
        """Reset training state."""

        self._total_step = 0
        self._total_epoch = 0
