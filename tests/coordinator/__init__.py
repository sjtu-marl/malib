import ray

from malib.utils.typing import TaskRequest, List


@ray.remote
class FakeCoordinator:
    def gen_simulation_task(self, task_request: TaskRequest, mathces: List):
        pass

    def gen_training_task(self):
        raise NotImplementedError

    def gen_rollout_task(self):
        raise NotImplementedError
