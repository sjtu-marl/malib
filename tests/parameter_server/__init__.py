import ray

from malib.backend.datapool.parameter_server import ParameterServer, TableStatus
from malib.utils.typing import ParameterDescription, Status


@ray.remote
class FakeParameterServer:
    def __init__(self):
        print("fake parameterserver created.")

    def start(self):
        return True

    def pull(self, parameter_desc: ParameterDescription, keep_return: bool = False):
        status = Status.NORMAL
        return status, parameter_desc

    def push(self, parameter_desc: ParameterDescription):
        status = TableStatus(False, Status.NORMAL)
        return status
