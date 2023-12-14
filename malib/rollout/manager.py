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

from typing import Dict, Tuple, Any, Callable, Set, List, Union

import traceback
import ray
import numpy as np

from ray.util import ActorPool

from malib.utils.logging import Logger
from malib.common.task import RolloutTask
from malib.common.manager import Manager
from malib.remote.interface import RemoteInterface
from malib.common.strategy_spec import StrategySpec
from malib.rollout.config import RolloutConfig
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
            continue
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
        stopping_conditions: Dict[str, Any],
        num_worker: int,
        group_info: Dict[str, Any],
        rollout_config: Union[RolloutConfig, Dict[str, Any]],
        env_desc: Dict[str, Any],
        log_dir: str,
        resource_config: Dict[str, Any] = None,
        ray_actor_namespace: str = "rollout_worker",
        verbose: bool = True,
    ):
        """Construct a manager for multiple rollout workers.

        Args:
            num_worker (int): Indicates how many rollout workers will be initialized.
            rollout_config (Dict[str, Any]): Runtime rollout configuration.
            env_desc (Dict[str, Any]): Environment description.
            log_dir (str): Log directory.
            resource_config (Dict[str, Any], optional): A dict that describes the resource config. Defaults to None.
            verbose (bool, optional): Enable logging or not. Defaults to True.
        """

        super().__init__(verbose=verbose, namespace=ray_actor_namespace)

        rollout_worker_cls = PBRolloutWorker
        worker_cls = rollout_worker_cls.as_remote(num_cpus=0, num_gpus=0)
        workers = []
        ready_check = []
        for i in range(num_worker):
            workers.append(
                worker_cls.options(
                    max_concurrency=100, namespace=self.namespace, name=f"actor_{i}"
                ).remote(
                    env_desc=env_desc,
                    agent_groups=group_info["agent_groups"],
                    rollout_config=RolloutConfig.from_raw(rollout_config),
                    log_dir=log_dir,
                    rollout_callback=None,
                    simulate_callback=None,
                    resource_config=resource_config,
                    verbose=verbose,
                )
            )
            ready_check.append(workers[-1].ready.remote())

        while len(ready_check):
            _, ready_check = ray.wait(ready_check, num_returns=1, timeout=1)

        self._workers: List[ray.ObjectRef] = workers
        self._actor_pool = ActorPool(self._workers)
        self._runtime_ids = tuple(group_info["agent_groups"].keys())
        self._group_info = group_info

        # FIXME(ming): deprecated
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

        return self._group_info["agent_groups"]

    @property
    def workers(self) -> List[RemoteInterface]:
        """Return a list of registered workers.

        Returns:
            List[RemoteInterface]: A list of workers.
        """

        return self._workers

    def submit(
        self, task: Union[Dict[str, Any], List[Dict[str, Any]]], wait: bool = False
    ) -> Any:
        """Submit a task to workers

        Args:
            task (Union[Dict[str, Any], List[Dict[str, Any]]]): Task description or a list of task description
            task_type (Any): Task type, should be an instance from TaskType
        """

        if isinstance(task, List):
            task = [RolloutTask.from_raw(e) for e in task]
        else:
            task = [RolloutTask.from_raw(task)]

        for _task in task:
            validate_strategy_specs(_task.strategy_specs)
            self._actor_pool.submit(
                lambda actor, _task: actor.rollout.remote(_task), _task
            )

        if wait:
            result_list = self.wait()
            return result_list
        else:
            return None

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
            Logger.error(traceback.format_exc())
            raise e

    def terminate(self):
        """Stop all remote workers"""

        super().terminate()

        for worker in self._workers:
            worker.close.remote()
            ray.kill(worker)
