import uuid
from functools import wraps
from abc import ABCMeta, abstractmethod
from malib.utils.logger import Logger

from malib.utils.typing import TaskDescription, TaskRequest, Sequence, Dict
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
        self._task_cache: Dict[str, Dict] = {}

    def pre_launching(self, init_config):
        pass

    def generate_task_id(self):
        return uuid.uuid4().hex

    @property
    def task_cache(self) -> Dict[str, Dict]:
        return self._task_cache

    @property
    def training_manager(self) -> TrainingManager:
        return self._training_manager

    @property
    def rollout_manager(self) -> RolloutWorkerManager:
        return self._rollout_worker_manager

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
