# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

"""
A `RolloutWorkerManager` contains a cluster of `RolloutWorker` (in the future version, each worker will be wrapped in a
subprocess). It is responsible for the resources management of worker instances, also statistics collections. Workers
will be assigned with rollout tasks sent from the `CoordinatorServer`.
"""

from typing import Dict, Tuple, Any, Callable, Set, List
from collections import defaultdict

import traceback
import ray
import numpy as np

from ray.util import ActorPool

from malib.common.manager import Manager
from malib.remote.interface import RemoteInterface
from malib.common.strategy_spec import StrategySpec
from malib.rollout.pb_rolloutworker import PBRolloutWorker


def validate_strategy_specs(specs: Dict[str, StrategySpec]):
    """Validate a dict of strategy specs that whether the prob list is legal.

    Args:
        specs (Dict[str, StrategySpec]): A dict of strategy spec.

    Raises:
        ValueError: Empty spec for some runtime id.
        ValueError: Give an empty prob list explicitly for some spec.
        ValueError: Summation of prob list is not close to 1.
    """

    for rid, spec in specs.items():
        if len(spec) < 1:
            raise ValueError(f"Empty spec for runtime_id={rid}")
        # check prob list
        expected_prob_list = spec.meta_data.get(
            "prob_list", [1 / len(spec)] * len(spec)
        )
        if expected_prob_list is None:
            raise ValueError(
                f"donot give an empty prob list explictly for runtime_id={rid}."
            )
        if not np.isclose(sum(expected_prob_list), 1.0):
            raise ValueError(
                f"The summation of prob list for runtime_id={rid} shoud be close to 1.: {expected_prob_list}."
            )


class RolloutWorkerManager(Manager):
    def __init__(
        self,
        experiment_tag: str,
        stopping_conditions: Dict[str, Any],
        num_worker: int,
        agent_mapping_func: Callable,
        rollout_config: Dict[str, Any],
        env_desc: Dict[str, Any],
        log_dir: str,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        """Construct a manager for multiple rollout workers.

        Args:
            experiment_tag (str): Experiment tag.
            num_worker (int): Indicates how many rollout workers will be initialized.
            agent_mapping_func (Callable): Agent mapping function, maps agents to runtime id.
            rollout_config (Dict[str, Any]): Runtime rollout configuration.
            env_desc (Dict[str, Any]): Environment description.
            log_dir (str): Log directory.
            resource_config (Dict[str, Any], optional): A dict that describes the resource config. Defaults to None.
            verbose (bool, optional): Enable logging or not. Defaults to True.
        """

        super().__init__(verbose=verbose)

        rollout_worker_cls = PBRolloutWorker
        worker_cls = rollout_worker_cls.as_remote(num_cpus=0, num_gpus=0)
        workers = []

        for i in range(num_worker):
            workers.append(
                worker_cls.options(max_concurrency=100).remote(
                    experiment_tag=experiment_tag,
                    env_desc=env_desc,
                    agent_mapping_func=agent_mapping_func,
                    rollout_config=rollout_config,
                    log_dir=log_dir,
                    rollout_callback=None,
                    simulate_callback=None,
                    resource_config=resource_config,
                    verbose=verbose,
                )
            )

        self._workers: List[ray.actor] = workers
        self._actor_pool = ActorPool(self._workers)

        agent_groups = defaultdict(lambda: set())
        for agent in env_desc["possible_agents"]:
            rid = agent_mapping_func(agent)
            agent_groups[rid].add(agent)
        self._runtime_ids = tuple(agent_groups.keys())
        self._agent_groups = dict(agent_groups)
        self.experiment_tag = experiment_tag

        assert (
            "rollout" in stopping_conditions
        ), f"Stopping conditions should contain `rollout`: {stopping_conditions}"
        self.stopping_conditions = stopping_conditions

    @property
    def runtime_ids(self) -> Tuple[str]:
        """A tuple of active runtime ids.

        Returns:
            Tuple[str]: A tuple of runtime ids.
        """

        return self._runtime_ids

    @property
    def agent_groups(self) -> Dict[str, Set]:
        """A dict of agent groups.

        Returns:
            Dict[str, Set]: A dict of set.
        """

        return self._agent_groups

    @property
    def workers(self) -> List[RemoteInterface]:
        """Return a list of registered workers.

        Returns:
            List[RemoteInterface]: A list of workers.
        """

        return self._workers

    def simulate(self, task_list):
        """Parse simulation task and dispatch it to available workers"""

        for task in task_list:
            self._actor_pool.submit(
                lambda actor, task: actor.simulate.remote(runtime_strategy_specs=task),
                task,
            )

    def rollout(self, task_list: List[Dict[str, Any]]) -> None:
        """Start rollout task without blocking.

        Args:
            task_list (List[Dict[str, Any]]): A list of task dict, keys include:
                - `strategy_specs`: a dict of strategy specs, mapping from runtime ids to specs.
                - `trainable_agents`: a list of trainable agents.

        """

        # validate all strategy specs here
        for task in task_list:
            validate_strategy_specs(task["strategy_specs"])

        while self._actor_pool.has_next():
            try:
                self._actor_pool.get_next(timeout=0)
            except TimeoutError:
                pass

        for task in task_list:
            self._actor_pool.submit(
                lambda actor, task: actor.rollout.remote(
                    runtime_strategy_specs=task["strategy_specs"],
                    stopping_conditions=self.stopping_conditions["rollout"],
                    trainable_agents=task["trainable_agents"],
                    data_entrypoints=task["data_entrypoints"],
                ),
                task,
            )

    def retrive_results(self):
        """Retrieve task results

        Raises:
            e: Any exceptions.

        Yields:
            Any: Task results.
        """

        try:
            while self._actor_pool.has_next():
                yield self._actor_pool.get_next()
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def terminate(self):
        """Stop all remote workers"""

        super().terminate()

        for worker in self._workers:
            worker.close.remote()
            ray.kill(worker)
