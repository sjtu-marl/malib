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

from threading import Thread

import time
import pytest
import numpy as np
import ray

from ray.util.queue import Queue
from readerwriterlock import rwlock

from malib.utils.episode import Episode
from malib.utils.tianshou_batch import Batch
from malib.utils.replay_buffer import ReplayBuffer
from malib.backend.offline_dataset_server import (
    OfflineDataset,
    write_table,
    read_table,
)


def is_actor_done(actor):
    if actor is None:
        return True
    done_ref = actor.__ray_terminate__.remote()
    done, not_done = ray.wait([done_ref], timeout=5)
    return len(not_done) == 0


def start_reader_thread(reader, n_round, read_size):
    try:
        for i in range(n_round):
            if not is_actor_done(reader.actor):
                batch_info = reader.get()
            else:
                break
    except Exception as e:
        print("done for actor has been terminated")


def start_writer_thread(writer, n_round, write_size):
    obs_array = np.random.random((write_size, 3))
    action_array = np.random.random((write_size, 2))
    rew_array = np.random.random(write_size)
    obs_next_array = np.random.random((write_size, 3))

    batch = Batch(
        {
            Episode.CUR_OBS: obs_array,
            Episode.ACTION: action_array,
            Episode.NEXT_OBS: obs_next_array,
            Episode.REWARD: rew_array,
        }
    )
    try:
        for _ in range(n_round):
            if is_actor_done(writer.actor):
                writer.put_nowait_batch([batch])
            else:
                break
    except Exception as e:
        print("done for actor has been terminated")


@pytest.mark.parametrize("read_size,write_size", [(64, 64), (64, 128), (128, 64)])
def test_datatable_read_and_write(read_size: int, write_size: int):
    if not ray.is_initialized():
        ray.init()

    buffer_size = 10000
    marker = rwlock.RWLockFair()
    writer = Queue(actor_options={"num_cpus": 0.1, "name": str(time.time())})
    reader = Queue(actor_options={"num_cpus": 0.1, "name": str(time.time())})
    buffer = ReplayBuffer(buffer_size)

    write_thread = Thread(target=write_table, args=(marker, buffer, writer))
    read_thread = Thread(target=read_table, args=(marker, buffer, read_size, reader))

    write_thread.start()
    read_thread.start()

    n_round = 1000
    reader_thread = Thread(
        target=start_reader_thread, args=(reader, n_round, read_size)
    )
    writer_thread = Thread(
        target=start_writer_thread, args=(writer, n_round, write_size)
    )

    reader_thread.start()
    writer_thread.start()

    reader_thread.join()
    writer_thread.join()

    reader.shutdown()
    writer.shutdown()

    read_thread.join()
    write_thread.join()

    ray.shutdown()


def test_offline_dataset():

    if not ray.is_initialized():
        ray.init()

    server = OfflineDataset(table_capacity=10000)
    server.start()

    # functionality test
    pname, pqueue = server.start_producer_pipe(name="test_offline_dataset")
    cname, cqueue = server.start_consumer_pipe(
        name="test_offline_dataset", batch_size=64
    )

    server.end_consumer_pipe(name=cname)
    server.end_producer_pipe(name=pname)

    ray.shutdown()
