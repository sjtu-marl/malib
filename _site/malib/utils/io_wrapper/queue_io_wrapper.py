from .base_io_wrapper import BaseIOWrapper
from multiprocessing import Queue


class QueueIOWrapper(BaseIOWrapper):
    def __init__(self, in_queue: Queue, out_queue: Queue):
        self._i = in_queue
        self._o = out_queue

    def write(self, object):
        self._o.put(object)

    def read(self):
        self._i.get()
