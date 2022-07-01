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

    def wait(self):
        collected_rets = []

        try:
            if isinstance(self.pending_tasks, List):
                while len(self.pending_tasks) > 0:
                    if self._force_stop:
                        self.terminate()
                        break
                    else:
                        dones, self.pending_tasks = ray.wait(self.pending_tasks)
                        collected_rets.extend(ray.get(dones))
            elif isinstance(self.pending_tasks, Generator):
                for task in self.pending_tasks:
                    if isinstance(task, list):
                        collected_rets.extend(task)
                    else:
                        collected_rets.append(task)
                    if self._force_stop:
                        self.terminate()
                        break
            else:
                raise ValueError("Unknow type: {}".format(self.pending_tasks))
        except Exception as e:
            traceback.print_exc()
            raise e

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
