from typing import Any, Dict, List
from concurrent import futures
import threading
import traceback
import ray

from readerwriterlock import rwlock

from malib.utils.logging import Logger
from malib.common.manager import Manager


class League:
    def __init__(
        self,
        training_manager: Manager,
        rollout_manager: Manager,
        inference_manager: Manager,
    ) -> None:
        self.training_manager = training_manager
        self.rollout_manager = rollout_manager
        self.inferenc_managfer = inference_manager
        self.flight_servers = []
        self.rw_lock = rwlock.RWLockFair()
        self.event = threading.Event()
        self.thread_pool = futures.ThreadPoolExecutor()

    def register_flight_server(self, flight_server_address: str):
        raise NotImplementedError

    def list_flight_servers(self) -> List[str]:
        raise NotImplementedError

    def _flight_server_check(self):
        while not self.event.is_set():
            with self.rw_lock.gen_rlock():
                for flight_server in self.flight_servers:
                    if not ray.util.check_connection(flight_server):
                        self.flight_servers.remove(flight_server)
            self.event.wait(10)

    def list_learners(self):
        return self.training_manager.workers()

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
                for result in self.training_manager.get_results():
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
        self.training_manager.terminate()
        self.rollout_manager.terminate()
