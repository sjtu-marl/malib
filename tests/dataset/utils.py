import ray

from collections import namedtuple

from malib.utils.typing import BufferDescription


Batch = namedtuple("Batch", "identity,data")


@ray.remote(num_cpus=0)
class FakeDataServer:
    def save(self, buffer_desc: BufferDescription):
        pass

    def get_producer_index(self, buffer_desc: BufferDescription):
        indices = list(range(buffer_desc.batch_size))
        return Batch(buffer_desc.identify, indices)
