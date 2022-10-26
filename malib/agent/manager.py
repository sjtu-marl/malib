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

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, List, Tuple, Union, Set, Sequence, Type
from collections import defaultdict

import os
import ray

from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.utils.exploitability import measure_exploitability
from malib.remote.interface import RemoteInterface
from malib.agent.agent_interface import AgentInterface
from malib.common.strategy_spec import StrategySpec
from malib.common.manager import Manager


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
        training_config: Dict[str, Any],
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
            agent_mapping_func (Callable[[AgentID], str]): The mapping function maps agent id to training interface id.
            training_config (Dict[str, Any]): Training configuration, for agent interface, keys include \
                `type`, `trainer_config` and `custom_config`.
            log_dir (str): Directory for logging.
            remote_mode (bool, Optional): Init agent interfaces as remote actor or not. Default is True.
        """

        super().__init__(verbose=verbose)

        resource_config = resource_config or DEFAULT_RESOURCE_CONFIG

        # interface config give the agent type used here and the group mapping if needed
        agent_groups = defaultdict(lambda: set())
        for agent in env_desc["possible_agents"]:
            rid = agent_mapping_func(agent)
            agent_groups[rid].add(agent)

        # FIXME(ming): resource configuration is not available now, will open in the next version
        if training_config["trainer_config"].get("use_cuda", False):
            num_gpus = 1 / len(agent_groups)
        else:
            num_gpus = 0.0
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        agent_cls = training_config["type"]
        # update num gpus
        resource_config["num_gpus"] = num_gpus
        agent_cls = agent_cls.as_remote(**resource_config).options(max_concurrency=10)
        interfaces: Dict[str, Union[AgentInterface, ray.ObjectRef]] = {}

        assert (
            "training" in stopping_conditions
        ), f"Stopping conditions should contains `training` stoppong conditions: {stopping_conditions}"

        for rid, agents in agent_groups.items():
            handler = agent_cls.remote if remote_mode else agent_cls
            interfaces[rid] = handler(
                experiment_tag=experiment_tag,
                runtime_id=rid,
                log_dir=f"{log_dir}/learner_{rid}",
                env_desc=env_desc,
                algorithms=algorithms,
                agent_mapping_func=agent_mapping_func,
                governed_agents=tuple(agents),
                trainer_config=training_config["trainer_config"],
                custom_config=training_config.get("custom_config"),
                verbose=verbose,
            )

        # ensure all interfaces have been started up
        if remote_mode:
            _ = ray.get([x.connect.remote() for x in interfaces.values()])

        self._agent_groups = agent_groups
        self._runtime_ids = tuple(self._agent_groups.keys())
        self._experiment_tag = experiment_tag
        self._env_description = env_desc
        self._training_config = training_config
        self._log_dir = log_dir
        self._agent_mapping_func = agent_mapping_func
        self._interfaces = interfaces
        self._remote_mode = remote_mode
        self._thread_pool = ThreadPoolExecutor(max_workers=len(interfaces))
        self._stopping_conditions = stopping_conditions

        Logger.info(
            f"training manager launched, {len(self._interfaces)} learner(s) created"
        )

    @property
    def agent_groups(self) -> Dict[str, Set[AgentID]]:
        """A dict describes the agent grouping, maps from runtime ids to agent sets.

        Returns:
            Dict[str, Set[AgentID]]: A dict of agent set.
        """

        return self._agent_groups

    @property
    def workers(self) -> List[RemoteInterface]:
        return list(self._interfaces.values())

    @property
    def runtime_ids(self) -> Tuple[str]:
        return self._runtime_ids

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
            interface_ids = list(self._interfaces.keys())

        assert isinstance(interface_ids, (List, Tuple, Set)), type(interface_ids)

        ns = dict.fromkeys(interface_ids, n) if isinstance(n, int) else n
        if self._remote_mode:
            strategy_spec_list: List[StrategySpec] = ray.get(
                [
                    self._interfaces[k].add_policies.remote(n=ns[k])
                    for k in interface_ids
                ]
            )
            strategy_spec_dict: Dict[str, StrategySpec] = dict(
                zip(interface_ids, strategy_spec_list)
            )
        else:
            strategy_spec_dict = {
                k: self._interfaces[k].add_policies(n=ns[k]) for k in interface_ids
            }

        return strategy_spec_dict

    def run(self, data_request_identifiers: Dict[str, str]):
        """Start training thread without blocking"""

        if self._remote_mode:
            for rid, interface in self._interfaces.items():
                self.pending_tasks.append(
                    interface.train.remote(
                        data_request_identifiers[rid],
                        self._stopping_conditions["training"],
                    )
                )
        else:
            for rid, interface in self._interfaces.items():
                self.pending_tasks.append(
                    self._thread_pool.submit(
                        interface.train,
                        data_request_identifiers[rid],
                        self._stopping_conditions["training"],
                    )
                )

    def retrive_results(self):
        while len(self.pending_tasks) > 0:
            dones, self.pending_tasks = ray.wait(self.pending_tasks)
            for done in ray.get(dones):
                yield done

    def terminate(self) -> None:
        """Terminate all training actors."""

        super().terminate()

        if self._remote_mode:
            for x in self._interfaces.values():
                ray.kill(x)

        self._thread_pool.shutdown()
        del self._interfaces

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
