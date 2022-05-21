import uuid
import ray


from malib.remote.interface import RemoteInterFace
from malib.utils.typing import Dict
from malib.rollout.manager import RolloutWorkerManager
from malib.agent.manager import TrainingManager


class BaseCoordinator(RemoteInterFace):
    def __init__(self):
        RemoteInterFace.__init__(self)
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
        return self._rollout_manager

    @rollout_manager.setter
    def rollout_manager(self, value):
        self._rollout_manager = value

    @training_manager.setter
    def training_manager(self, value):
        self._training_manager = value

    # def aggregate(self, **kwargs):
    #     pass

    # def dispatch(self, **kwargs):
    #     pass
