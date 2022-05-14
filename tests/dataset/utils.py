from typing import Any

import threading
import ray

from collections import namedtuple

from malib.utils.typing import BufferDescription, Dict, AgentID, Status


Batch = namedtuple("Batch", "identity,data")


@ray.remote
class FakeDataServer:
    def __init__(self) -> None:
        self.push_lock = False
        self.pull_lock = True
        self.threading_lock = threading.Lock()
        print("fake dataset server created.")

    def start(self):
        return True

    def save(self, buffer_desc: BufferDescription):
        pass

    def create_table(self, name: str, reverb_server_kwargs: Dict[str, Any]):
        pass
