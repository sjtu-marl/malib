import logging
import os
import sys
import traceback
import threading
import asyncio
import time
import traceback
import numpy as np
import torch
import ray

from collections import namedtuple, defaultdict
from concurrent.futures import ThreadPoolExecutor
from readerwriterlock import rwlock
from torch._C import dtype

from malib import settings
from malib.utils.errors import OversampleError
from malib.utils.general import BufferDict, iter_many_dicts_recursively
from malib.utils.logger import Log, Logger
from malib.utils.typing import (
    BufferDescription,
    PolicyID,
    AgentID,
    Dict,
    List,
    Any,
    Union,
    Tuple,
    Status,
)

import threading
import pickle as pkl


def _gen_table_name(env_id, main_id, pid):
    res = f"{env_id}"
    if main_id:
        if isinstance(main_id, List):
            main_id = "_".join(sorted(main_id))
        res += f"_{main_id}"
    if pid:
        if isinstance(pid, List):
            pid = "_".join(sorted(pid))
        res += f"_{pid}"
    return res


DATASET_TABLE_NAME_GEN = _gen_table_name
Batch = namedtuple("Batch", "identity, data")


class Empty(Exception):
    pass


class Full(Exception):
    pass


def _start_loop(loop: asyncio.BaseEventLoop):
    asyncio.set_event_loop(loop)
    if not loop.is_running():
        loop.run_forever()


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


class _QueueActor:
    def __init__(self, maxsize, event_loop):
        self.maxsize = maxsize
        self.queue = asyncio.Queue(self.maxsize, loop=event_loop)

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    async def put(self, item, timeout=None):
        try:
            await asyncio.wait_for(self.queue.put(item), timeout)
        except asyncio.TimeoutError:
            raise Full

    async def get(self, timeout=None):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            raise Empty

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    def put_nowait_batch(self, items):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
            raise Full(
                f"Cannot add {len(items)} items to queue of size "
                f"{self.qsize()} and maxsize {self.maxsize}."
            )
        for item in items:
            self.queue.put_nowait(item)

    def get_nowait(self):
        return self.queue.get_nowait()

    def get_nowait_batch(self, num_items):
        if num_items > self.qsize():
            raise Empty(
                f"Cannot get {num_items} items from queue of size " f"{self.qsize()}."
            )
        return [self.queue.get_nowait() for _ in range(num_items)]

    def shutdown(self):
        self.queue = None


class Table:
    def __init__(
        self,
        capacity: int,
        fragment_length: int,
        data_shapes: Dict[AgentID, Dict[str, Tuple]] = None,
        data_dtypes: Dict[AgentID, Dict[str, Tuple]] = None,
        sample_start_size: int = 0,
        event_loop: asyncio.BaseEventLoop = None,
        name: str = "",
        mode: str = "queue",
    ):
        """One table for one episode."""
        self._name = name
        self._threading_lock = threading.Lock()
        self._rwlock = rwlock.RWLockFairD()
        self._consumer_queue = None
        self._producer_queue = None
        self._is_fixed = False
        self._sample_start_size = sample_start_size
        self._size = 0
        self._flag = 0
        self._capacity = capacity
        self._fragment_length = fragment_length
        self._data_shapes = data_shapes
        self._mode = mode

        if mode == "queue":
            self._consumer_queue = _QueueActor(maxsize=capacity, event_loop=event_loop)
            self._producer_queue = _QueueActor(maxsize=capacity, event_loop=event_loop)
            # ready index
            self._producer_queue.put_nowait_batch([i for i in range(capacity)])
        else:
            self._consumer_queue = None
            self._producer_queue = None

        # build episode
        if data_shapes is not None:
            self._buffer = BufferDict()
            for agent, _dshapes in data_shapes.items():
                # if agent not in keys:
                #     continue
                t = BufferDict()
                for dk, dshape in _dshapes.items():
                    # XXX(ming): use fragment length for RNN?
                    # XXX(ziyu): For the case that need a total episode with each timestep in order,
                    # we add fragment_length + 1 to the shape,
                    # '+1' is because truncated mode to get the bootstrap value.
                    t[dk] = np.zeros(
                        (capacity,)
                        + ((fragment_length + 1,) if self._fragment_length > 0 else ())
                        + dshape,
                        dtype=data_dtypes[agent][dk],
                    )
                self._buffer[agent] = t
        else:
            self._buffer = None

    @property
    def is_fixed(self):
        return self._is_fixed

    @property
    def is_multi_agent(self) -> bool:
        # always true
        return True  # len(self.buffer)

    @property
    def buffer(self) -> BufferDict:
        return self._buffer

    @property
    def size(self):
        return self._size

    @property
    def flag(self):
        return self._flag

    @property
    def capacity(self):
        return self._capacity

    @property
    def name(self):
        return self._name

    def build_buffer_from_samples(self, sample: Dict):
        self._buffer = BufferDict()
        for agent, _buff in sample.items():
            t = BufferDict()
            for dk, v in _buff.items():
                t[dk] = np.zeros((self.capacity,) + v.shape[1:], dtype=v.dtype)
            self._buffer[agent] = t

    def sample_activated(self) -> bool:
        return self._consumer_queue.size() >= self._sample_start_size

    def fix_table(self):
        self._is_fixed = True
        if self._mode == "queue":
            self._producer_queue.shutdown()
            self._consumer_queue.shutdown()

    def get_producer_index(self, buffer_size: int) -> Union[List[int], None]:
        buffer_size = min(self._producer_queue.qsize(), buffer_size)
        if buffer_size <= 0:
            return None
        else:
            return self._producer_queue.get_nowait_batch(int(buffer_size))

    def get_consumer_index(self, buffer_size: int) -> Union[List[int], None]:
        buffer_size = min(self._consumer_queue.qsize(), buffer_size)
        if buffer_size <= 0:
            return None
        else:
            return self._consumer_queue.get_nowait_batch(int(buffer_size))

    def free_consumer_index(self, indices: List[int]):
        self._producer_queue.put_nowait_batch(indices)

    def free_producer_index(self, indices: List[int]):
        self._consumer_queue.put_nowait_batch(indices)

    @staticmethod
    def gen_table_name(*args, **kwargs):
        return DATASET_TABLE_NAME_GEN(*args, **kwargs)

    def insert(
        self, data: List[Dict[str, Any]], indices: List[int] = None, size: int = None
    ):
        assert isinstance(data, List), type(data)
        if self.buffer is None:
            self.build_buffer_from_samples(data[0])

        if indices is None:
            # generate indices
            indices = np.arange(self._flag, self._flag + size) % self._capacity

        shuffle_idx = np.arange(len(indices))
        np.random.shuffle(shuffle_idx)
        for d_list, k, value_list in iter_many_dicts_recursively(*data):
            head_d = d_list[0]
            batch_sizes = [v.shape[0] for v in value_list]
            merged_shape = (sum(batch_sizes),) + value_list[0].shape[1:]
            _placeholder = np.zeros(merged_shape, dtype=head_d[k].dtype)
            index = 0
            for batch_size, value in zip(batch_sizes, value_list):
                _placeholder[index : index + batch_size] = value[:]
                index += batch_size
            assert len(_placeholder) >= len(indices), (
                len(_placeholder),
                len(indices),
                _placeholder.shape,
                k,
                value_list[0].shape,
                len(value_list),
            )
            head_d[k] = _placeholder[shuffle_idx]

        # assert indices is not None, "indices: {}".format(indices)
        self._buffer.set_data(indices, data[0])
        self._size += len(indices)
        self._size = min(self._size, self._capacity)
        self._flag = (self._flag + len(indices)) % self._capacity

    def sample(self, indices: List[int] = None, size: int = None) -> Dict[str, Any]:
        if indices is None:
            indices = np.random.choice(self.size, size)
        return self._buffer.index(indices)

    @staticmethod
    def _save_helper_func(obj, fp, candidate_name=""):
        if os.path.isdir(fp):
            try:
                os.makedirs(fp)
            except:
                pass
            tfp = os.path.join(fp, candidate_name + ".tpkl")
        else:
            paths = os.path.split(fp)[0]
            try:
                os.makedirs(paths)
            except:
                pass
            tfp = fp + ".tpkl"
        with open(tfp, "wb") as f:
            pkl.dump(obj, f, protocol=settings.PICKLE_PROTOCOL_VER)

    def dump(self, fp, name=None):
        if name is None:
            name = self._name
        with self._threading_lock:
            serial_dict = {
                "fragment_length": self._fragment_length,
                "multi_agent": self.is_multi_agent,
                "sample_start_size": self._sample_start_size,
                "data_shapes": self._data_shapes,
                "data": self._buffer,
                "name": self._name,
            }
            self._save_helper_func(serial_dict, fp, name)

    @classmethod
    def load(cls, fp, event_loop=None):
        with open(fp, "rb") as f:
            serial_dict = pkl.load(f)

        buffer = serial_dict["data"]
        dtypes = {}
        for agent, agent_data in buffer.items():
            agent_dtypes = {}
            for cname, cdata in agent_data.items():
                agent_dtypes[cname] = cdata.dtype
            dtypes[agent] = agent_dtypes

        table = Table(
            capacity=buffer.capacity,
            fragment_length=serial_dict["fragment_length"],
            data_shapes=serial_dict["data_shapes"],
            data_dtypes=dtypes,
            sample_start_size=serial_dict["sample_start_size"],
            event_loop=event_loop,
            name=serial_dict.get("name", ""),
        )
        table._buffer = buffer
        return table

    def to_csv(self, fp):
        def _dump_episode(fname, episode: BufferDict):
            class _InternalColumnGenerator:
                def __init__(self, column_data_dict):
                    self.idx = 0
                    self.data = column_data_dict
                    self.columns = column_data_dict.keys()
                    self.length = len(next(iter(column_data_dict.values())))

                def getlines(self):
                    column_info = "/".join([str(col) for col in self.columns]) + "\n"
                    yield column_info
                    while self.idx < self.length:
                        line = []
                        for c in self.columns:
                            line.append(str(self.data[c][self.idx].tolist()))
                        line = "/".join(line) + "\n"
                        self.idx += 1
                        yield line

            lines = _InternalColumnGenerator(episode).getlines()
            with open(fname, "w") as f:
                f.writelines(lines)

        with self._threading_lock:
            try:
                os.makedirs(fp)
            except:
                pass

            assert self.is_multi_agent
            for aid in self._buffer.keys():
                episode = self._buffer[aid]
                _dump_episode(os.path.join(fp, str(aid)), episode)


@ray.remote
class OfflineDataset:
    def __init__(self, dataset_config: Dict[str, Any]):
        self._episode_capacity = dataset_config.get(
            "episode_capacity", settings.DEFAULT_EPISODE_CAPACITY
        )
        self._fragment_length = dataset_config.get("fragment_length")
        self._learning_start = dataset_config.get("learning_start", 64)
        self._tables: Dict[str, Table] = dict()
        self._threading_lock = threading.Lock()
        self._threading_pool = ThreadPoolExecutor()

        loop = get_or_create_eventloop()
        self.event_loop = loop
        self.event_thread = threading.Thread(target=_start_loop, args=(loop,))
        self.event_thread.setDaemon(True)
        self.event_thread.start()

        # parse init tasks
        init_job_config = dataset_config.get("init_job", {})
        if init_job_config.get("load_when_start"):
            path = init_job_config.get("path")
            if path:
                self.load(path)

        # # Read-only proxies for external offline dataset
        # external_resource_config = dataset_config.get("extern")
        # self.external_proxy: List[ExternalDataset] = []
        # if external_resource_config:
        #     for external_config, sample_rate in zip(
        #         external_resource_config["links"],
        #         external_resource_config["sample_rates"],
        #     ):
        #         if not external_config["write"]:
        #             dataset = ExternalReadOnlyDataset(
        #                 name=external_config["name"],
        #                 path=external_config["path"],
        #                 sample_rate=sample_rate,
        #             )
        #             self.external_proxy.append(dataset)
        #         else:
        #             raise NotImplementedError(
        #                 "External writable dataset is not supported"
        #             )

        # quitting job
        quit_job_config = dataset_config.get("quit_job", {})
        self.dump_when_closed = quit_job_config.get("dump_when_closed")
        self.dump_path = quit_job_config.get("path")
        Logger.info(
            "dataset server initialized with (table_capacity={} table_learning_start={})".format(
                self._episode_capacity, self._learning_start
            )
        )

    def lock(self, lock_type: str, desc: Dict[AgentID, BufferDescription]) -> str:
        """Lock table ready to push or pull and return the table status."""

        env_id = list(desc.values())[0].env_id
        main_ids = sorted(list(desc.keys()))
        table_name = Table.gen_table_name(
            env_id=env_id,
            main_id=main_ids,
            pid=[desc[aid].policy_id for aid in main_ids],
        )
        # check it is multi-agent or not
        # self.check_table(table_name, None, is_multi_agent=len(main_ids) > 1)
        # table = self._tables[table_name]
        # status = table.lock_push_pull(lock_type)
        # return status
        if table_name in self._tables:
            return Status.SUCCESS
        else:
            return Status.FAILED

    def unlock(self, lock_type: str, desc: Dict[AgentID, BufferDescription]):
        env_id = list(desc.values())[0].env_id
        main_ids = sorted(list(desc.keys()))
        table_name = Table.gen_table_name(
            env_id=env_id,
            main_id=main_ids,
            pid=[desc[aid].policy_id for aid in main_ids],
        )
        # self.check_table(table_name, None, is_multi_agent=len(main_ids) > 1)
        # table = self._tables[table_name]
        # status = table.unlock_push_pull(lock_type)
        # return status
        if table_name in self._tables:
            return Status.SUCCESS
        else:
            return Status.FAILED

    def create_table(self, buffer_desc: BufferDescription, ignore=True):
        name = Table.gen_table_name(
            env_id=buffer_desc.env_id,
            main_id=buffer_desc.agent_id,
            pid=buffer_desc.policy_id,
        )

        if name in self._tables:
            if not ignore:
                raise Warning("Repeated table definite: {}".format(name))
            # return None
            pass
        else:
            self._tables[name] = Table(
                self._episode_capacity,
                self._fragment_length,
                # buffer_desc.data_shapes,
                sample_start_size=self._learning_start,
                event_loop=self.event_loop,
                name=name,
            )
            Logger.info("created data table: {}".format(name))

    def get_consumer_index(
        self, buffer_desc: BufferDescription
    ) -> Union[List[int], None]:
        """Before saving, get index"""

        try:
            table_name = Table.gen_table_name(
                env_id=buffer_desc.env_id,
                main_id=buffer_desc.agent_id,
                pid=buffer_desc.policy_id,
            )
            table = self._tables[table_name]
            indices = table.get_consumer_index(buffer_desc.batch_size)
        except KeyError:
            Logger.warn("table {} not ready yet for indexing".format(table_name))
            indices = None

        return Batch(buffer_desc.identify, indices)

    def get_producer_index(
        self, buffer_desc: BufferDescription
    ) -> Union[List[int], None]:
        """Before saving, get index"""

        try:
            table_name = Table.gen_table_name(
                env_id=buffer_desc.env_id,
                main_id=buffer_desc.agent_id,
                pid=buffer_desc.policy_id,
            )
            table = self._tables[table_name]
            indices = table.get_producer_index(buffer_desc.batch_size)
        except KeyError:
            # Logger.warn("table {} not ready yet for indexing".format(table_name))
            indices = None

        return Batch(buffer_desc.identify, indices)

    def save(self, buffer_desc: BufferDescription):
        table_name = Table.gen_table_name(
            env_id=buffer_desc.env_id,
            main_id=buffer_desc.agent_id,
            pid=buffer_desc.policy_id,
        )
        table = self._tables[table_name]
        table.insert(buffer_desc.data, indices=buffer_desc.indices)
        table.free_producer_index(buffer_desc.indices)

    @Log.method_timer(enable=settings.PROFILING)
    def load_from_dataset(
        self,
        file: str,
        env_id: str,
        policy_id: PolicyID,
        agent_id: AgentID,
    ):
        """
        Expect the dataset to be in the form of List[ Dict[str, Any] ]
        """

        # FIXME(ming): check its functionality
        # with open(file, "rb") as f:
        #     dataset = pkl.load(file=f)
        #     keys = set()
        #     for batch in dataset:
        #         keys = keys.union(batch.keys())

        #     table_size = len(dataset)
        #     table_name = DATASET_TABLE_NAME_GEN(
        #         env_id=env_id,
        #         main_id=agent_id,
        #         pid=policy_id,
        #     )
        #     if self._tables.get(table_name, None) is None:
        #         self._tables[table_name] = Episode(
        #             env_id, policy_id, other_columns=None
        #         )

        #     for batch in dataset:
        #         assert isinstance(batch, Dict)
        #         self._tables[table_name].insert(**batch)
        raise NotImplementedError

    # @Log.method_timer(enable=settings.PROFILING)
    def load(self, path, mode="replace") -> List[Dict[str, str]]:
        table_descs = []
        with self._threading_lock:
            for fn in os.listdir(path):
                if fn.endswith(".tpkl"):
                    conflict_callback_required = False
                    table = Table.load(os.path.join(path, fn))
                    victim_table = None

                    if table.name in self._tables.keys() and mode.lower() == "replace":
                        victim_table = self._tables[table.name]
                        Logger.debug(
                            f"Conflicts in loading table {table.name}, act as replacing"
                        )
                        # try_lock_status = victim_table.lock_push_pull("push")
                        # if try_lock_status != Status.NORMAL:
                        #     Logger.error(
                        #         f"Try to replace an occupied table {victim_table.name}"
                        #     )
                        conflict_callback_required = True

                    self._tables[table.name] = table
                    table_descs.append(
                        {
                            "name": table.name,
                            "size": table.size,
                            "capacity": table.capacity,
                        }
                    )

                    if conflict_callback_required:
                        # victim_table.unlock_push_pull("push")
                        victim_table.fix_table()
                        del victim_table

            return table_descs

    # @Log.method_timer(enable=settings.PROFILING)
    def dump(self, path, table_names=None, as_csv=False):
        with self._threading_lock:
            if table_names is None:
                table_names = list(self._tables.keys())
            elif isinstance(table_names, str):
                table_names = [table_names]

            # Locking required tables
            status = dict.fromkeys(table_names, Status.FAILED)
            for tn in table_names:
                table = self._tables[tn]
                # status[tn] = table.lock_push_pull("push")
                status[tn] = Status.SUCCESS

            # Check locking status
            f = open("ds.log", "w")
            for tn, lock_status in status.items():
                print(tn)
                f.write(f"Table {tn} lock status {lock_status}")
                if lock_status == Status.FAILED:
                    Logger.info(f"Failed to lock the table {tn}, skip the dumping.")
                    continue
                if not as_csv:
                    self._tables[tn].dump(os.path.join(path, tn))
                else:
                    self._tables[tn].to_csv(os.path.join(path, tn))
                # self._tables[tn].unlock_push_pull("push")
            f.close()
            return status

    # @Log.method_timer(enable=settings.PROFILING)
    def sample(self, buffer_desc: BufferDescription) -> Tuple[Batch, str]:
        """Sample data from the top for training, default is random sample from sub batches.
        :param BufferDesc buffer_desc: Description of sampling a buffer.
            used to index the buffer slot
        :return: a tuple of samples and information
        """

        # generate idxes from idxes manager
        res = None
        info = "OK"
        # with Log.timer(log=settings.PROFILING, logger=Logger):
        try:
            table_name = Table.gen_table_name(
                env_id=buffer_desc.env_id,
                main_id=buffer_desc.agent_id,
                pid=buffer_desc.policy_id,
            )
            table = self._tables[table_name]
            res = table.sample(indices=buffer_desc.indices)
            table.free_consumer_index(buffer_desc.indices)
        except KeyError:
            info = "table {} has not been created yet".format(table_name)
        except OverflowError:
            info = "no enough size for table {} yet".format(table_name)
        res = Batch(identity=buffer_desc.agent_id, data=res)
        return res, info

    def shutdown(self):
        status = None
        if self.dump_when_closed:
            Logger.info("Begin OfflineDataset dumping.")
            status = self.dump(self.dump_path)
        Logger.info("Server terminated.")
        return status


# class ExternalDataset:
#     def __init__(self, name, path, sample_rate=0.5):
#         self._name = name
#         if os.path.isabs(path):
#             self._path = path
#         else:
#             self._path = os.path.join(settings.BASE_DIR, path)
#         self._sample_rate = sample_rate

#     def sample(self):
#         raise NotImplementedError

#     def save(self):
#         raise NotImplementedError


# class ExternalReadOnlyDataset(ExternalDataset):
#     def __init__(
#         self, name, path, sample_rate=0.5, mapping_func=lambda x: x, binary=True
#     ):
#         super().__init__(name, path, sample_rate=sample_rate)

#         self._tables: Dict[str, Table] = {}
#         for fn in os.listdir(self._path):
#             if fn.endswith(".tpkl"):
#                 table = Table.load(os.path.join(self._path, fn))
#                 self._tables[table.name] = table

#     def sample(self, buffer_desc: BufferDescription):
#         info = f"{self._name}(external, read-only): OK"
#         try:
#             # NOTE(zbzhu): maybe we do not care which policy sampled the (expert) data
#             table_name = Table.gen_table_name(
#                 env_id=buffer_desc.env_id,
#                 main_id=buffer_desc.agent_id,
#                 pid=None,
#                 # pid=buffer_desc.policy_id,
#             )
#             table = self._tables[table_name]
#             res = table.sample(size=self._sample_rate * buffer_desc.batch_size)
#         except KeyError as e:
#             info = f"data table `{table_name}` has not been created {list(self._tables.keys())}"
#             res = None
#         except OversampleError as e:
#             info = f"No enough data: table_size={table.size} batch_size={buffer_desc.batch_size} table_name={table_name}"
#             res = None
#         except Exception as e:
#             print(traceback.format_exc())
#             res = None
#             info = "others"
#         return res, info

# def save(self, agent_episodes: Dict[AgentID, Episode], wait_for_ready: bool = True):
#     raise NotImplementedError
