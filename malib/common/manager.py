from abc import abstractmethod, ABC

import ray


class Manager(ABC):
    @abstractmethod
    def __init__(self):
        self._force_stop = False
        self.pending_tasks = []

    def force_stop(self):
        self._force_stop = True

    def wait(self):
        while len(self.pending_tasks) > 0:
            if self._force_stop:
                self.terminate()
            else:
                dones, self.pending_tasks = ray.wait(self._pending_tasks)

    @abstractmethod
    def terminate(self):
        """Resource recall"""
