from threading import Thread

import pytest
import numpy as np
import ray

from ray.util.queue import Queue
from readerwriterlock import rwlock

from malib.utils.episode import Episode
from malib.utils.tianshou_batch import Batch
from malib.utils.replay_buffer import ReplayBuffer
from malib.backend.offline_dataset_server import OfflineDataset, write_table, read_table


def start_reader_thread(reader, n_round, read_size):
    for i in range(n_round):
        if reader.actor is not None:
            batch_info = reader.get()
        else:
            break


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
    for _ in range(n_round):
        if writer.actor is not None:
            writer.put_nowait_batch([batch])
        else:
            break


@pytest.mark.parametrize("read_size,write_size", [(64, 64), (64, 128), (128, 64)])
def test_datatable_read_and_write(read_size: int, write_size: int):
    buffer_size = 10000
    marker = rwlock.RWLockFair()
    writer = Queue(actor_options={"num_cpus": 0})
    reader = Queue(actor_options={"num_cpus": 0})
    buffer = ReplayBuffer(buffer_size)

    threads = []
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

    print("threads for reader/writer closed")

    # kill queue
    writer.shutdown()
    reader.shutdown()

    print("queues have been killed")

    read_thread.join()
    print("read thread has been closed")
    write_thread.join()
    print("write_thread has been closed")


def test_offline_dataset():

    server = OfflineDataset(table_capacity=10000)
    server.start()

    # functionality test
    pname, pqueue = server.start_producer_pipe(name="test_offline_dataset")
    cname, cqueue = server.start_consumer_pipe(
        name="test_offline_dataset", batch_size=64
    )

    server.end_consumer_pipe(name=cname)
    server.end_producer_pipe(name=pname)
