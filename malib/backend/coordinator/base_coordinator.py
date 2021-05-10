from abc import ABCMeta, abstractmethod
from typing import Sequence

from malib.utils.typing import TaskDescription, TaskRequest


class TaskGraph:
    def __init__(self):
        pass

    def add_task_node(self, task_description: TaskDescription):
        pass

    def predict(self, node_ids: Sequence[str]):
        pass

    def subgraph(self):
        pass


class BaseCoordinator(metaclass=ABCMeta):
    def __init__(self):
        self.task_graph = TaskGraph()

    @abstractmethod
    def push(self, task_request: TaskRequest):
        """ Accept task results from workers/actors """

        # check whether all dependencies
        pass

    @abstractmethod
    def pull(self, **kwargs):
        pass

    def aggregate(self, **kwargs):
        pass

    def dispatch(self, **kwargs):
        pass
