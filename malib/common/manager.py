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

import traceback
from typing import List, Generator, Any
from abc import abstractmethod, ABC

import ray

from malib.utils.logging import Logger
from malib.remote.interface import RemoteInterface


class Manager(ABC):
    @abstractmethod
    def __init__(self, verbose: bool):
        self._force_stop = False
        self.pending_tasks = []
        self.verbose = verbose

    def is_running(self):
        return len(self.pending_tasks) > 0

    def force_stop(self):
        self._force_stop = True

    @property
    def workers(self) -> List[RemoteInterface]:
        raise NotImplementedError

    def retrive_results(self):
        raise NotImplementedError

    def wait(self) -> List[Any]:
        """Wait workers to be terminated, and retrieve the executed results.

        Returns:
            List[Any]: A list of results.
        """

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
