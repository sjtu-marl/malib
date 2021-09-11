from abc import ABCMeta, abstractmethod
from typing import Sequence

from malib.utils.typing import TaskDescription, TaskRequest
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager


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
        self._training_manager: TrainingManager = None
        self._rollout_manager: RolloutWorkerManager = None

    def pre_launching(self, init_config):
        pass

    @staticmethod
    def task_handler_register(cls):
        from functools import wraps

        print("Registering")

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(*args, **kwargs)

            setattr(cls, func.__name__, func)
            return func

        return decorator

    @property
    def training_manger(self) -> TrainingManager:
        return self._training_manager

    @property
    def rollout_manager(self) -> RolloutWorkerManager:
        return self._rollout_manager

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
