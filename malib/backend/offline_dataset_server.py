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

import threading
import traceback
from typing import Dict, Any, Tuple, Union, List

# import reverb

from malib.remote.interface import RemoteInterface
from malib.utils.typing import PolicyID, AgentID
from malib.utils.logging import Logger


# class ReverbDataset(RemoteInterface):
#     def __init__(self, table_capacity: int):
#         self.servers: Dict[str, reverb.Server] = {}
#         self.tb_params_list_dict: Dict[str, Dict[str, Any]] = {}
#         self.capacity = table_capacity
#         self.lock = threading.Lock()

#     def start(self):
#         Logger.info("Dataset server started")

#     def get_port(self, table_name: str):
#         with self.lock:
#             if table_name in self.servers:
#                 return self.servers[table_name].port
#             else:
#                 return None

#     def get_client_kwargs(self, table_name: str) -> Dict[str, Any]:
#         """Retrieve reverb client kwargs.

#         Args:
#             table_name (str): _description_

#         Returns:
#             Dict[str, Any]: _description_
#         """

#         with self.lock:
#             if table_name in self.servers:
#                 server = self.servers[table_name]
#                 tb_params_list = self.tb_params_list_dict[table_name]
#                 return {"address": server.port, "tb_params_list": tb_params_list}
#             else:
#                 return {"address": None, "tb_params_list": None}

#     def create_table(self, name: str, reverb_server_kwargs: Dict[str, Any]):
#         """Create sub reverb server.

#         Args:
#             name (str): Sub server name.
#             reverb_server_kwargs (Dict[str, Any]): Reverb server kwargs, keys including:
#             - `port`: str, optional, the gRPC port string.
#             - `checkpointer`: reverb.platform.default.CheckpointerBase, optional.
#             - `tb_params_list`: list of dict, required, for constructing tables.

#         Note:
#             for each item in `tb_params_list`, the dict should include the folloing keys:
#             - `name`:
#             - `sampler`:
#             - `remover`:
#             - `max_size`: int, if not given, will use default `self.capacity`.
#             - `rate_limiter`:
#             - `max_times_sampled`:
#             - `extension`:
#             - `signature`:
#         """

#         print("offline dataset ready to open a server: {}".format(reverb_server_kwargs))

#         try:
#             with self.lock:
#                 if name not in self.servers:
#                     port = reverb_server_kwargs.get("port", None)
#                     checkpointer = reverb_server_kwargs.get("checkpointer", None)
#                     tb_params_list = reverb_server_kwargs["tb_params_list"]
#                     self.servers[name] = reverb.Server(
#                         tables=[
#                             reverb.Table(
#                                 name=tb_params["name"],
#                                 sampler=tb_params.get(
#                                     "sampler", reverb.selectors.Uniform()
#                                 ),
#                                 remover=tb_params.get(
#                                     "remover", reverb.selectors.Fifo()
#                                 ),
#                                 max_size=tb_params.get("max_size", self.capacity),
#                                 rate_limiter=tb_params.get(
#                                     "rate_limiter", reverb.rate_limiters.MinSize(1)
#                                 ),
#                                 max_times_sampled=tb_params.get("max_times_sampled", 0),
#                                 extensions=tb_params.get("extension", ()),
#                                 signature=tb_params.get("signature", None),
#                             )
#                             for tb_params in tb_params_list
#                         ],
#                         port=port,
#                         checkpointer=checkpointer,
#                     )
#                     self.tb_params_list_dict[name] = tb_params_list
#         except Exception:
#             traceback.print_exc()

#     def load_from_dataset(
#         self,
#         file: str,
#         env_id: str,
#         policy_id: PolicyID,
#         agent_id: AgentID,
#     ):
#         """
#         Expect the dataset to be in the form of List[ Dict[str, Any] ]
#         """
#         raise NotImplementedError


import time

from concurrent.futures import ThreadPoolExecutor
from ray.util.queue import Queue

import numpy as np
import ray

from readerwriterlock import rwlock
from malib.utils.tianshou_batch import Batch
from malib.utils.tianshou_replay import ReplayBuffer


def write_table(marker: rwlock.RWLockFair, buffer: ReplayBuffer, writer: Queue):
    wlock = marker.gen_wlock()
    while True:
        try:
            batch: Union[Batch, List[Batch]] = writer.get()
            with wlock:
                if isinstance(batch, List):
                    batch = batch[0]
                buffer.add_episode(batch)
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
            print(f"create buffer for name={name}")
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
            print("waiting for consumer pipe ...")
            time.sleep(1)
        self.thread_pool.submit(
            read_table, self.markers[name], self.buffers[name], batch_size, queue
        )
        return queue_id, queue

    def end_consumer_pipe(self, name: str):
        if name in self.reader_queues:
            queue = self.reader_queues.pop(name)
            ray.kill(queue)
