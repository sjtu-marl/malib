import uuid
import ray

from abc import ABCMeta, abstractmethod

from malib.utils.typing import TaskRequest, Dict
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager


class BaseCoordinator(metaclass=ABCMeta):
    def __init__(self):
        self._training_manager: TrainingManager = None
        self._rollout_manager: RolloutWorkerManager = None
        self._task_cache: Dict[str, Dict] = {}

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization"""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

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
        return self._rollout_manager

    # def aggregate(self, **kwargs):
    #     pass

    # def dispatch(self, **kwargs):
    #     pass
