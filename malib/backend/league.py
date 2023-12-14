from typing import Any, Dict, List
from concurrent import futures
import threading
import traceback
import ray

from readerwriterlock import rwlock

from malib.utils.logging import Logger
from malib.common.manager import Manager
from malib.common.task import Task, RolloutTask, OptimizationTask


class League:
    def __init__(
        self,
        learner_manager: Manager,
        rollout_manager: Manager,
        inference_manager: Manager,
    ) -> None:
        self.learner_manager = learner_manager
        self.rollout_manager = rollout_manager
        self.inferenc_managfer = inference_manager
        self.rw_lock = rwlock.RWLockFair()
        self.event = threading.Event()
        self.thread_pool = futures.ThreadPoolExecutor()

    def list_learners(self):
        return self.learner_manager.workers()

    def submit(self, task_desc: Task, wait: bool = False):
        if isinstance(task_desc, RolloutTask):
            res = self.rollout_manager.submit(task_desc, wait)
        elif isinstance(task_desc, OptimizationTask):
            res = self.learner_manager.submit(task_desc, wait)
        else:
            raise ValueError(f"Unexpected task type: {isinstance(task_desc)}")
        return res

    def list_rollout_workers(self):
        return self.rollout_manager.workers()

    def list_inference_clients(self):
        return self.inferenc_managfer.workers()

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve results from rollout and training manager.

        Returns:
            Dict[str, Dict[str, Any]]: A dict of results, which contains rollout and training results.
        """

        rollout_results = []
        training_results = []

        try:
            while True:
                for result in self.rollout_manager.get_results():
                    rollout_results.append(result)
                for result in self.learner_manager.get_results():
                    training_results.append(result)
        except KeyboardInterrupt:
            Logger.info("Keyboard interruption was detected, recalling resources ...")
        except RuntimeError:
            Logger.error(traceback.format_exc())
        except Exception:
            Logger.error(traceback.format_exc())

        return {"rollout": rollout_results, "training": training_results}

    def terminate(self):
        self.event.set()
        self.thread_pool.shutdown()
        self.learner_manager.terminate()
        self.rollout_manager.terminate()
