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

from typing import (
    Dict,
    Any,
    Callable,
    List,
    Tuple,
    Union,
    Set,
    Sequence,
    Type,
    Generator,
)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError

import os
import traceback
import ray
from malib.common.task import OptimizationTask

from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.utils.exploitability import measure_exploitability
from malib.remote.interface import RemoteInterface
from malib.agent.agent_interface import AgentInterface
from malib.common.strategy_spec import StrategySpec
from malib.common.manager import Manager
from malib.common.training_config import TrainingConfig


DEFAULT_RESOURCE_CONFIG = dict(
    num_cpus=None, num_gpus=None, memory=None, object_store_memory=None, resources=None
)


class TrainingManager(Manager):
    def __init__(
        self,
        experiment_tag: str,
        stopping_conditions: Dict[str, Any],
        algorithms: Dict[str, Any],
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable[[AgentID], str],
        group_info: Dict[str, Any],
        training_config: Union[Dict[str, Any], TrainingConfig],
        log_dir: str,
        remote_mode: bool = True,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        """Create an TrainingManager instance which is responsible for the multi agent training
        tasks execution and rollout task requests sending.

        Args:
            experiment_tag (str): Experiment identifier, for data tracking.
            algorithms (Dict[str, Any]): The algorithms configuration candidates.
            env_desc (Dict[str, Any]): The description for environment generation.
            interface_config (Dict[str, Any]): Configuration for agent training inferece construction, keys include \
                `type` and `custom_config`, a dict.
            agent_mapping_func (Callable[[AgentID], str]): The mapping function maps environment agent id to training agent (`agent_interface`) id.
            training_config (Dict[str, Any]): Training configuration, for agent interface, keys include \
                `type`, `trainer_config` and `custom_config`.
            log_dir (str): Directory for logging.
            remote_mode (bool, Optional): Init learners as remote actor or not. Default is True.
        """

        super().__init__(verbose=verbose)

        resource_config = resource_config or DEFAULT_RESOURCE_CONFIG
        training_config = TrainingConfig.from_raw(training_config)

        # interface config give the agent type used here and the group mapping if needed

        # FIXME(ming): resource configuration is not available now, will open in the next version
        if training_config.trainer_config.get("use_cuda", False):
            num_gpus = 1 / len(group_info["agent_groups"])
        else:
            num_gpus = 0.0
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        learner_cls = training_config.learner_type
        # update num gpus
        resource_config["num_gpus"] = num_gpus
        learner_cls = learner_cls.as_remote(**resource_config).options(
            max_concurrency=10
        )
        learners: Dict[str, Union[AgentInterface, ray.ObjectRef]] = {}

        assert (
            "training" in stopping_conditions
        ), f"Stopping conditions should contains `training` stoppong conditions: {stopping_conditions}"

        for rid, agents in group_info["agent_groups"].items():
            _cls = learner_cls.remote if remote_mode else learner_cls
            learners[rid] = _cls(
                experiment_tag=experiment_tag,
                runtime_id=rid,
                log_dir=f"{log_dir}/learner_{rid}",
                env_desc=env_desc,
                algorithms=algorithms,
                agent_mapping_func=agent_mapping_func,
                governed_agents=tuple(agents),
                trainer_config=training_config.trainer_config,
                custom_config=training_config.custom_config,
                verbose=verbose,
            )

        # ensure all interfaces have been started up
        if remote_mode:
            _ = ray.get([x.connect.remote() for x in learners.values()])

        # TODO(ming): collect data entrypoints from learners
        self._group_info = group_info
        self._runtime_ids = tuple(self._agent_groups.keys())
        self._experiment_tag = experiment_tag
        self._env_description = env_desc
        self._training_config = training_config
        self._log_dir = log_dir
        self._agent_mapping_func = agent_mapping_func
        self._learners = learners
        self._remote_mode = remote_mode
        self._thread_pool = ThreadPoolExecutor(max_workers=len(learners))
        self._stopping_conditions = stopping_conditions

        Logger.info(
            f"training manager launched, {len(self._learners)} learner(s) created"
        )

    @property
    def agent_groups(self) -> Dict[str, Set[AgentID]]:
        """A dict describes the agent grouping, maps from runtime ids to agent sets.

        Returns:
            Dict[str, Set[AgentID]]: A dict of agent set.
        """

        return self._group_info["agent_groups"]

    @property
    def get_data_entrypoints(self) -> Dict[str, str]:
        """Return a dict of data entrypoints, maps from runtime ids to data entrypoints.

        Returns:
            Dict[str, str]: A dict of data entrypoints.
        """

        return {rid: rid for rid in self._runtime_ids}

    @property
    def workers(self) -> List[RemoteInterface]:
        """A list of learner instance

        Returns:
            List[RemoteInterface]: A list of learner instance
        """

        return list(self._learners.values())

    @property
    def runtime_ids(self) -> Tuple[str]:
        """Return a tuple of learner ids

        Returns:
            Tuple[str]: A tuple of string as leqrner ids
        """

        return self._runtime_ids

    def get_data_entrypoint_mapping(self) -> Dict[AgentID, str]:
        raise NotImplementedError

    def add_policies(
        self, interface_ids: Sequence[str] = None, n: Union[int, Dict[str, int]] = 1
    ) -> Dict[str, Type[StrategySpec]]:
        """Notify interface `interface_id` add `n` policies and return the newest strategy spec.

        Args:
            interface_ids (Sequence[str]): Registered agent interface id.
            n (int, optional): Indicates how many policies will be added.

        Returns:
            Dict[str, Type[StrategySpec]]: A dict of strategy specs, maps from runtime ids to strategy specs.
        """

        if interface_ids is None:
            interface_ids = list(self._learners.keys())

        assert isinstance(interface_ids, (List, Tuple, Set)), type(interface_ids)

        policy_nums = dict.fromkeys(interface_ids, n) if isinstance(n, int) else n

        if self._remote_mode:
            strategy_spec_list: List[StrategySpec] = ray.get(
                [
                    self._learners[k].add_policies.remote(n=policy_nums[k])
                    for k in interface_ids
                ]
            )
            strategy_spec_dict: Dict[str, StrategySpec] = dict(
                zip(interface_ids, strategy_spec_list)
            )
        else:
            strategy_spec_dict = {
                k: self._learners[k].add_policies(n=policy_nums[k])
                for k in interface_ids
            }

        return strategy_spec_dict

    def submit(self, task: OptimizationTask):
        """Submit a training task, the manager will distribute it to the corresponding learners.

        Args:
            task (OptimizationTask): A task description.
        """

        # retrieve learners with active agents
        for aid in task.active_agents:
            rid = self._agent_mapping_func(aid)
            if rid not in self._learners:
                raise RuntimeError(f"Agent {aid} is not registered in training manager")
            else:
                learner = self._learners[rid]
                if self._remote_mode:
                    ray_task = learner.train.remote(task)
                    self.pending_tasks.append(ray_task)
                else:
                    raise NotImplementedError

    def retrive_results(self) -> Generator:
        """Return a generator of results.

        Yields:
            Generator: A generator for task results.
        """

        if self._remote_mode:
            while len(self.pending_tasks) > 0:
                dones, self.pending_tasks = ray.wait(self.pending_tasks)
                for done in ray.get(dones):
                    yield done
        else:
            for task in self.pending_tasks:
                assert isinstance(task, Future)
                try:
                    if task.done():
                        yield task.result(timeout=10)
                except TimeoutError:
                    Logger.error(
                        f"Retrieving results of training task is timeout: {traceback.format_exc()}"
                    )
                except CancelledError:
                    Logger.error(
                        f"Try to retrieve results of a cancelled task: {traceback.format_exc()}"
                    )
                except Exception:
                    Logger.error(traceback.format_exc())

    def terminate(self) -> None:
        """Terminate all training actors."""

        super().terminate()

        if self._remote_mode:
            for x in self._learners.values():
                ray.kill(x)

        self._thread_pool.shutdown()
        del self._learners

    def get_exp(self, policy_distribution):
        """Compute exploitability"""

        # XXX(ming): may we migrate this function to (PSRO) evaluator
        populations = {}
        for aid, agent in self._agents.items():
            populations[aid] = ray.get(agent.get_policies.remote())
        nashconv, _ = measure_exploitability(
            self._env_description["config"]["env_id"],
            populations=populations,
            policy_mixture_dict=policy_distribution,
        )

        return nashconv
