"""
A `RolloutWorkerManager` contains a cluster of `RolloutWorker` (in the future version, each worker will be wrapped in a
subprocess). It is responsible for the resources management of worker instances, also statistics collections. Workers
will be assigned with rollout tasks sent from the `CoordinatorServer`.
"""

from typing import Dict, Tuple, Any, Callable
import hashlib
import time

import ray

from ray.util import ActorPool

from malib.common.manager import Manager
from malib.rollout.rollout_worker import RolloutWorker


def _get_worker_hash_idx(idx):
    hash_coding = hashlib.md5()
    hash_coding.update(bytes(f"worker-{idx}-{time.time()}", "utf-8"))
    return hash_coding.hexdigest()


class RolloutWorkerManager(Manager):
    def __init__(
        self,
        num_worker: int,
        agent_mapping_func: Callable,
        rollout_configs: Dict[str, Any],
        env_desc: Dict[str, Any],
        exp_cfg: Dict[str, Any],
    ):
        """Create a rollout worker manager. A rollout worker manager is responsible for a group of rollout workers. For
        each rollout/simulation tasks dispatched from `CoordinatorServer`, it will be assigned to an idle worker which
        executes the tasks in parallel.

        :param Dict[str,Any] rollout_config: Rollout configuration
        :param Dict[str,Any] env_desc: Environment description, to create environment instances.
        :param Dict[str,Any] exp_cfg: Experiment description.
        """

        super().__init__()

        rollout_worker_cls = RolloutWorker
        worker_cls = rollout_worker_cls.as_remote(
            num_cpus=None,
            num_gpus=None,
            memory=None,
            object_store_memory=None,
            resources=None,
        )
        workers = {}

        for i in range(num_worker):
            worker_idx = _get_worker_hash_idx(i)
            workers[worker_idx] = worker_cls.options(max_concurrency=100).remote(
                worker_index=worker_idx,
                env_desc=env_desc,
                agent_mapping_func=agent_mapping_func,
                experiment_config=exp_cfg,
                runtime_configs=rollout_configs,
            )

        self._workers: Dict[str, ray.actor] = workers
        self._actor_pool = ActorPool(self._workers)

    def simulate(self, task_list):
        """Parse simulation task and dispatch it to available workers"""

        self._actor_pool.map(
            lambda actor, task: actor.simulate.remote(
                runtime_strategy_specs=task.strategy_specs,
                stopping_conditions=task.stopping_conditions,
                trainable_agents=task.trainable_agents,
            ),
            task_list,
        )

    def rollout(self, task_list) -> None:
        """Start rollout task without blocking"""

        self._actor_pool.map(
            lambda actor, task: actor.rollout.remote(
                runtime_strategy_specs=task.strategy_specs,
                stopping_conditions=task.stopping_conditions,
                trainable_agents=task.trainable_agents,
            ),
            task_list,
        )

    def wait(self):
        while not self._actor_pool.has_free():
            if self._force_stop:
                self.terminate()
                break

    def terminate(self):
        """Stop all remote workers"""

        for worker in self._workers.values():
            worker.close.remote()
            ray.kill(worker)
