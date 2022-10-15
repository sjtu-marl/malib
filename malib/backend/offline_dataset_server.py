# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from readerwriterlock import rwlock

import traceback
import time

import numpy as np
import ray

from ray.util.queue import Queue

from malib.remote.interface import RemoteInterface
from malib.utils.logging import Logger
from malib.utils.tianshou_batch import Batch
from malib.utils.replay_buffer import ReplayBuffer


def write_table(marker: rwlock.RWLockFair, buffer: ReplayBuffer, writer: Queue):
    wlock = marker.gen_wlock()
    while True:
        try:
            batches: Union[Batch, List[Batch]] = writer.get()
            with wlock:
                if not isinstance(batches, List):
                    batches = [batches]
                for e in batches:
                    buffer.add_batch(e)
        except Exception as e:
            Logger.warning(f"writer queue dead for: {traceback.format_exc()}")
            break


def read_table(
    marker: rwlock.RWLockFair, buffer: ReplayBuffer, batch_size: int, reader: Queue
):
    rlock = marker.gen_rlock()
    while True:
        try:
            with rlock:
                if len(buffer) >= batch_size:
                    batch, indices = buffer.sample(batch_size)
                else:
                    batch, indices = [], np.array([], int)
            reader.put_nowait((batch, indices))
        except Exception as e:
            Logger.warning(f"reader queue dead for: {traceback.format_exc()}")
            break


class OfflineDataset(RemoteInterface):
    def __init__(self, table_capacity: int, max_consumer_size: int = 1024) -> None:
        self.tb_capacity = table_capacity
        self.reader_queues: Dict[str, Queue] = {}
        self.writer_queues: Dict[str, Queue] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}
        self.markers: Dict[str, rwlock.RWLockFair] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_consumer_size)

    def start(self):
        Logger.info("Dataset server started")

    def start_producer_pipe(
        self,
        name: str,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs,
    ) -> Tuple[str, Queue]:
        if name not in self.buffers:
            buffer = ReplayBuffer(
                size=self.tb_capacity,
                stack_num=stack_num,
                ignore_obs_next=ignore_obs_next,
                save_only_last_obs=save_only_last_obs,
                sample_avail=sample_avail,
                **kwargs,
            )
            marker = rwlock.RWLockFair()
            writer = Queue(actor_options={"num_cpus": 0})

            self.buffers[name] = buffer
            self.markers[name] = marker
            self.writer_queues[name] = writer
            self.thread_pool.submit(write_table, marker, buffer, writer)

        return name, self.writer_queues[name]

    def end_producer_pipe(self, name: str):
        if name in self.writer_queues:
            queue = self.writer_queues.pop(name)
            ray.kill(queue)

    def start_consumer_pipe(self, name: str, batch_size: int) -> Tuple[str, Queue]:
        queue_id = f"{name}_{time.time()}"
        queue = Queue(actor_options={"num_cpus": 0})
        self.reader_queues[queue_id] = queue
        # make sure that the buffer is ready
        while name not in self.buffers:
            time.sleep(1)
        self.thread_pool.submit(
            read_table, self.markers[name], self.buffers[name], batch_size, queue
        )
        return queue_id, queue

    def end_consumer_pipe(self, name: str):
        if name in self.reader_queues:
            queue = self.reader_queues.pop(name)
            ray.kill(queue)
