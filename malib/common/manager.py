import traceback
from typing import List, Generator
from abc import abstractmethod, ABC

import ray

from malib.utils.logging import Logger
from malib.remote.interface import RemoteInterface


class Manager(ABC):
    @abstractmethod
    def __init__(self):
        self._force_stop = False
        self.pending_tasks = []

    def is_running(self):
        return len(self.pending_tasks) > 0

    def force_stop(self):
        self._force_stop = True

    @property
    def workers(self) -> List[RemoteInterface]:
        raise NotImplementedError

    def retrive_results(self):
        raise NotImplementedError

    def wait(self):
        collected_rets = []
        for res in self.retrive_results():
            collected_rets.append(res)
        return collected_rets

    def cancel_pending_tasks(self):
        """Cancle all running tasks."""

        rets = None

        try:
            ray.get([w.stop_pending_tasks.remote() for w in self.workers])
            rets = self.wait()
        except Exception as e:
            Logger.warning(
                "tray to cancel pending tasks, but met some exception: {}".format(e)
            )
        finally:
            self.pending_tasks = []

        return rets

    @abstractmethod
    def terminate(self):
        """Resource recall"""

        self.cancel_pending_tasks()
