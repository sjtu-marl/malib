import threading
import ray

from collections import namedtuple

from malib.utils.typing import BufferDescription, Dict, AgentID, Status


Batch = namedtuple("Batch", "identity,data")


@ray.remote(num_cpus=0)
class FakeDataServer:
    def __init__(self) -> None:
        self.push_lock = False
        self.pull_lock = True
        self.threading_lock = threading.Lock()

    def save(self, buffer_desc: BufferDescription):
        pass

    def get_producer_index(self, buffer_desc: BufferDescription):
        indices = list(range(buffer_desc.batch_size))
        return Batch(buffer_desc.identify, indices)

    def get_consumer_index(self, buffer_desc: BufferDescription):
        indices = list(range(buffer_desc.batch_size))
        return Batch(buffer_desc.identify, indices)

    def sample(self, buffer_desc: BufferDescription):
        res = (None,)
        info = 200
        return Batch(identity=buffer_desc.agent_id, data=res), info

    def lock(self, lock_type: str, desc: Dict[AgentID, BufferDescription]) -> str:
        """Lock table ready to push or pull and return the table status."""

        # lock type could be: push or pull
        # push has been locked, cannot lock pull
        with self.threading_lock:
            status = Status.SUCCESS
            if self.push_lock == False and lock_type == "pull":
                self.pull_lock = True
            elif self.pull_lock == False and lock_type == "push":
                self.push_lock = True
            else:
                status = Status.FAILED

        return status

    def unlock(self, lock_type: str, desc: Dict[AgentID, BufferDescription]):
        status = Status.SUCCESS
        with self.threading_lock:
            if lock_type == "pull" and self.pull_lock:
                self.pull_lock = False
            elif lock_type == "push" and self.push_lock:
                self.push_lock = False
            else:
                status = Status.FAILED
        return status
