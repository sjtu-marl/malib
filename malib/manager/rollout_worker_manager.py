"""
A `RolloutWorkerManager` contains a cluster of `RolloutWorker` (in the future version, each worker will be wrapped in a
subprocess). It is responsible for the resources management of worker instances, also statistics collections. Workers
will be assigned with rollout tasks sent from the `CoordinatorServer`.
"""

import hashlib
import os
import time

import psutil
import ray
import logging


from malib import settings
from malib.rollout.rollout_worker import RolloutWorker
from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    Status,
    Tuple,
    Dict,
    Any,
)
from malib.utils.logger import get_logger, Logger


def _get_worker_hash_idx(idx):
    hash_coding = hashlib.md5()
    hash_coding.update(bytes(f"worker-{idx}-{time.time()}", "utf-8"))
    return hash_coding.hexdigest()


class RolloutWorkerManager:
    def __init__(
        self,
        rollout_config: Dict[str, Any],
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

        self._workers: Dict[str, ray.actor] = {}
        self._config = rollout_config
        self._env_desc = env_desc
        self._metric_type = rollout_config["metric_type"]

        worker_num = rollout_config["worker_num"]
        rollout_worker_cls = RolloutWorker

        worker_cls = rollout_worker_cls.as_remote(
            num_cpus=None,
            num_gpus=None,
            memory=None,
            object_store_memory=None,
            resources=None,
        )

        for i in range(worker_num):
            worker_idx = _get_worker_hash_idx(i)
            self._workers[worker_idx] = worker_cls.options(max_concurrency=100).remote(
                worker_index=worker_idx,
                env_desc=self._env_desc,
                metric_type=self._metric_type,
                remote=True,
                save=rollout_config.get("save_model", False),
                # parallel_num: the size of actor pool for rollout and simulation
                parallel_num=rollout_config["num_episodes"]
                // rollout_config["num_env_per_worker"],
                exp_cfg=exp_cfg,
            )

        Logger.info(
            f"RolloutWorker manager launched, {len(self._workers)} rollout worker(s) alives."
        )

    def retrieve_information(self, task_request: TaskRequest) -> TaskRequest:
        """Retrieve information from other agent interface. Default do nothing and return the original task request.

        :param TaskRequest task_request: A task request from `CoordinatorServer`.
        :return: A task request
        """

        return task_request

    def get_idle_worker(self, test: bool = False) -> Tuple[str, RolloutWorker]:
        """Wait until an idle worker is available.

        :return: A tuple of worker index and worker.
        """

        status = Status.FAILED
        worker_idx, worker = None, None
        while status == Status.FAILED:
            for idx, t in self._workers.items():
                wstatus = ray.get(t.get_status.remote())
                if wstatus == Status.IDLE:
                    status = ray.get(t.set_status.remote(Status.LOCKED))
                if status == Status.SUCCESS:
                    worker_idx = idx
                    worker = t
                    break
        return worker_idx, worker

    def simulate(self, task_desc: TaskDescription, worker_idx=None):
        """Parse simulation task and dispatch it to available workers"""

        Logger.debug(
            f"got simulation task from handler: {task_desc.content.agent_involve_info.training_handler}"
        )
        worker_idx, worker = self.get_idle_worker()
        worker.simulation.remote(task_desc)

    def rollout(self, task_desc: TaskDescription, test: bool = False) -> None:
        """Parse rollout task and dispatch it to available worker.

        :param TaskDescription task_desc: A task description.
        :return: None
        """

        # split into several sub tasks rollout
        worker_idx, worker = self.get_idle_worker(test=test)
        worker.rollout.remote(task_desc)

    def terminate(self):
        """Stop all remote workers"""

        for worker in self._workers.values():
            worker.close.remote()
            worker.stop.remote()
            worker.__ray_terminate__.remote()
