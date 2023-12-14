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
from concurrent.futures import ThreadPoolExecutor

import os
import ray
from malib.common.task import OptimizationTask

from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.utils.exploitability import measure_exploitability
from malib.remote.interface import RemoteInterface
from malib.common.strategy_spec import StrategySpec
from malib.common.manager import Manager
from malib.learner.config import LearnerConfig
from malib.rl.config import Algorithm


DEFAULT_RESOURCE_CONFIG = dict(
    num_cpus=None, num_gpus=None, memory=None, resources=None
)


class LearnerManager(Manager):
    def __init__(
        self,
        stopping_conditions: Dict[str, Any],
        algorithm: Algorithm,
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable[[AgentID], str],
        group_info: Dict[str, Any],
        learner_config: LearnerConfig,
        log_dir: str,
        resource_config: Dict[str, Any] = None,
        ray_actor_namespace: str = "learner",
        verbose: bool = True,
    ):
        """Create an LearnerManager instance which is responsible for the multi agent training
        tasks execution and rollout task requests sending.

        Args:
            experiment_tag (str): Experiment identifier, for data tracking.
            algorithm (Dict[str, Any]): The algorithms configuration candidates.
            env_desc (Dict[str, Any]): The description for environment generation.
            interface_config (Dict[str, Any]): Configuration for agent training inferece construction, keys include \
                `type` and `custom_config`, a dict.
            agent_mapping_func (Callable[[AgentID], str]): The mapping function maps environment agent id to training agent (`agent_interface`) id.
            training_config (Dict[str, Any]): Training configuration, for agent interface, keys include \
                `type`, `trainer_config` and `custom_config`.
            log_dir (str): Directory for logging.
        """

        super().__init__(verbose=verbose, namespace=ray_actor_namespace)

        resource_config = resource_config or DEFAULT_RESOURCE_CONFIG
        learner_config = LearnerConfig.from_raw(learner_config)

        # interface config give the agent type used here and the group mapping if needed

        # FIXME(ming): resource configuration is not available now, will turn-on in the next version
        if algorithm.trainer_config.get("use_cuda", False):
            num_gpus = 1 / len(group_info["agent_groups"])
        else:
            num_gpus = 0.0
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        learner_cls = learner_config.learner_type
        # update num gpus
        resource_config["num_gpus"] = num_gpus
        learner_cls = learner_cls.as_remote(**resource_config)
        learners: Dict[str, ray.ObjectRef] = {}

        ready_check = []

        for rid, agents in group_info["agent_groups"].items():
            agents = tuple(agents)
            learners[rid] = learner_cls.options(
                name=f"learner_{rid}", max_concurrency=10, namespace=self.namespace
            ).remote(
                runtime_id=rid,
                log_dir=f"{log_dir}/learner_{rid}",
                observation_space=group_info["observation_space"][rid],
                action_space=group_info["action_space"][rid],
                algorithm=algorithm,
                governed_agents=agents,
                custom_config=learner_config.custom_config,
                feature_handler_gen=learner_config.feature_handler_meta_gen(
                    env_desc, agents[0]
                ),
                verbose=verbose,
            )
            ready_check.append(learners[rid].ready.remote())

        # ensure all interfaces have been started up
        while len(ready_check):
            _, ready_check = ray.wait(ready_check, num_returns=1, timeout=1)

        Logger.info("All Learners are ready for accepting new tasks.")
        data_entrypoints = ray.get(
            [x.get_data_entrypoint.remote() for x in learners.values()]
        )
        self._data_entrypoints = dict(zip(learners.keys(), data_entrypoints))
        self._learner_entrypoints = dict(
            zip(
                learners.keys(),
                [f"{self.namespace}:learner_{rid}" for rid in learners.keys()],
            )
        )

        # TODO(ming): collect data entrypoints from learners
        self._group_info = group_info
        self._runtime_ids = tuple(group_info["agent_groups"].keys())
        self._env_description = env_desc
        self._learner_config = learner_config
        self._log_dir = log_dir
        self._agent_mapping_func = agent_mapping_func
        self._learners = learners
        self._thread_pool = ThreadPoolExecutor(max_workers=len(learners))
        # FIXME(ming): deprecated
        self._stopping_conditions = stopping_conditions

        # init strategy spec
        self.add_policies()

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
    def data_entrypoints(self) -> Dict[str, str]:
        """Return a dict of data entrypoints, maps from runtime ids to data entrypoints.

        Returns:
            Dict[str, str]: A dict of data entrypoints.
        """

        return self._data_entrypoints

    @property
    def learner_entrypoints(self) -> Dict[str, str]:
        """Return a mapping from runtime ids to learner entrypoints.

        Returns:
            Dict[str, str]: A dict of learner entrypoints.
        """

        return self._learner_entrypoints

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

    def get_strategy_specs(self) -> Dict[str, StrategySpec]:
        values = ray.get(
            [v.get_strategy_spec.remote() for v in self._learners.values()]
        )
        return dict(zip(self._learners.keys(), values))

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

        strategy_spec_list: List[StrategySpec] = ray.get(
            [self._learners[k].get_strategy_spec.remote() for k in interface_ids]
        )
        strategy_spec_dict: Dict[str, StrategySpec] = dict(
            zip(interface_ids, strategy_spec_list)
        )

        return strategy_spec_dict

    def submit(self, task: OptimizationTask, wait: bool = False):
        """Submit a training task, the manager will distribute it to the corresponding learners.

        Args:
            task (OptimizationTask): A task description.
        """

        # retrieve learners with active agents
        rids = (
            list(self._learners.keys())
            if task.active_agents is None
            else [self._agent_mapping_func(aid) for aid in task.active_agents]
        )

        for rid in rids:
            learner = self._learners[rid]
            ray_task = learner.train.remote(task)
            self.pending_tasks.append(ray_task)
        if wait:
            result_list = self.wait()
            return result_list
        else:
            return None

    def retrive_results(self) -> Generator:
        """Return a generator of results.

        Yields:
            Generator: A generator for task results.
        """

        while len(self.pending_tasks):
            dones, self.pending_tasks = ray.wait(self.pending_tasks)
            for done in ray.get(dones):
                yield done

    def terminate(self) -> None:
        """Terminate all training actors."""

        super().terminate()

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
