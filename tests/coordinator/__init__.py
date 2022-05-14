import ray
from malib import agent
from malib.utils.logger import Logger

from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    List,
    TaskType,
    TrainingDescription,
)


@ray.remote
class FakeCoordinator:
    def __init__(self):
        print("fake coordinator created.")

    def start(self):
        return True

    def gen_simulation_task(self, task_request: TaskRequest, mathces: List):
        pass

    def gen_training_task(self):
        return None

    def gen_rollout_task(self):
        raise NotImplementedError

    def request(self, task_request: TaskRequest):
        pass
