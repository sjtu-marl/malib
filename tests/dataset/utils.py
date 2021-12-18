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

    def get_consumer_index(self, buffer_desc: BufferDescription):
        indices = list(range(buffer_desc.batch_size))
        return Batch(buffer_desc.identify, indices)

    def sample(self, buffer_desc: BufferDescription):
        res = (None,)
        info = 200
        return Batch(identity=buffer_desc.agent_id, data=res), info
