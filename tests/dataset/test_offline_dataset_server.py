import asyncio
from asyncio.events import AbstractEventLoop
from genericpath import exists
import os
import operator
import pytest
import ray
import numpy as np
import pickle as pkl

from functools import reduce

from malib.utils.typing import BufferDescription, List, Any, Dict, Status, Tuple
from malib.utils.episode import EpisodeKey
from malib.backend.datapool.offline_dataset_server import (
    BufferDict,
    OfflineDataset,
    Table,
    get_or_create_eventloop,
)

def test_get_or_create_eventloop():
    asyncio.set_event_loop(None)
    default_constructed_loop = get_or_create_eventloop()
    assert isinstance(default_constructed_loop, AbstractEventLoop)
    default_constructed_loop.close()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    assert isinstance(get_or_create_eventloop(), AbstractEventLoop)
    assert (get_or_create_eventloop() == loop)
    loop.close()
    asyncio.set_event_loop(None)
    get_or_create_eventloop()


@pytest.mark.parametrize(
    "keys,data_shapes,data_dtypes",
    [
        (
            [EpisodeKey.CUR_OBS, EpisodeKey.REWARD, EpisodeKey.DONE],
            [(3, 4), (), ()],
            [int, int, int],
        )
    ],
)
class TestBufferDict:
    @pytest.fixture(autouse=True)
    def _init(self, keys, data_shapes, data_dtypes):
        capacity = 100
        self.data_dtypes = data_dtypes
        self.capacity = capacity
        self.keys = keys
        self.data_shapes = data_shapes
        self.buffer = BufferDict()
        for k, shape, dtype in zip(keys, data_shapes, data_dtypes):
            self.buffer[k] = np.zeros((capacity,) + shape, dtype=dtype)

    def test_insert_and_sample(self):
        repeat = 10
        for _ in range(repeat):
            data = {}
            indices = np.random.choice(self.capacity, 10, replace=False)
            for k, shape, dtype in zip(self.keys, self.data_shapes, self.data_dtypes):
                full_shape = (10,) + shape
                data[k] = (
                    np.arange(reduce(operator.mul, full_shape))
                    .reshape(full_shape)
                    .astype(dtype)
                )
            self.buffer.set_data(indices, data)
            sample = self.buffer.index(indices)
            for k, d in sample.items():
                assert d.dtype == data[k].dtype
                assert d.shape == data[k].shape, (d.shape, data[k].shape)
                assert np.sum(np.abs(data[k] - d)) == 0
                # assert np.array_equal(d, data[k]), (k, d, data[k])


@pytest.mark.parametrize(
    "keys,capacity,fragment_length,shared_data_shapes,dtypes,sample_start_size",
    [
        (
            [1, 2],
            1000,
            0,
            {EpisodeKey.CUR_OBS: (3, 4), EpisodeKey.REWARD: (), EpisodeKey.DONE: ()},
            {
                EpisodeKey.CUR_OBS: np.float16,
                EpisodeKey.REWARD: np.float16,
                EpisodeKey.DONE: np.float16,
            },
            0,
        )
    ],
)
class TestTable:
    @pytest.fixture(autouse=True)
    def _init(
        self,
        keys: List[Any],
        capacity: int,
        fragment_length: int,
        shared_data_shapes: Dict[str, Tuple],
        dtypes: Dict[str, Any],
        sample_start_size: int,
        event_loop=None,
        mode: str = "queue",
    ):
        data_shapes = {k: shared_data_shapes for k in keys}
        dtypes = {k: dtypes for k in keys}
        self.table = Table(
            capacity,
            fragment_length,
            data_shapes,
            data_dtypes=dtypes,
            sample_start_size=sample_start_size,
            event_loop=event_loop,
            mode=mode,
        )

        assert 1 in self.table.buffer
        assert 2 in self.table.buffer

    def test_insert_and_sample(self):
        data_shapes = self.table._data_shapes
        # create samples from data shapes
        res = []
        sizes = [10, 20, 15]
        # no replace
        indices = np.random.choice(self.table.capacity, sum(sizes), replace=False)
        temp_for_check = {
            k: {_k: [] for _k in list(v.keys())} for k, v in data_shapes.items()
        }

        for i in range(3):
            main_dict = {}
            for main_key, shape_dict in data_shapes.items():
                _data = {}
                for k, shape in shape_dict.items():
                    full_shape = (sizes[i],) + shape
                    _data[k] = (
                        np.arange(reduce(operator.mul, full_shape))
                        .reshape(full_shape)
                        .astype(np.float16)
                    )
                    temp_for_check[main_key][k].append(_data[k])
                main_dict[main_key] = _data
            res.append(main_dict)
        self.table.insert(res, indices)

        assert self.table.size == 45

        # concate and sort
        for main_key, v in temp_for_check.items():
            # sort by axis=0
            for k, _v in v.items():
                temp_for_check[main_key][k] = np.sort(np.concatenate(_v), axis=0)

        # sample with indices
        data: BufferDict = self.table.sample(indices=indices)
        # check
        for k, _data in data.items():
            for _k, v in _data.items():
                assert isinstance(v, np.ndarray)
                _v = np.sort(v, axis=0)
                assert np.array_equal(_v, temp_for_check[k][_k]), (
                    k,
                    _k,
                    _v,
                    temp_for_check[k][_k],
                )

        # sample with size
        data: BufferDict = self.table.sample(size=sum(sizes))

    def test_queue_mode(self):
        def _size_check():
            assert (0 <= self.table._producer_queue.qsize() <= self.table.capacity)
            assert (0 <= self.table._consumer_queue.qsize() <= self.table.capacity)
            assert (self.table._producer_queue.qsize() + self.table._consumer_queue.qsize() == self.table.capacity)

        _size_check()
        cur_p_queue_size = self.table._producer_queue.qsize()
        cur_c_queue_size = self.table._consumer_queue.qsize()
        
        assert (self.table._producer_queue.full())
        assert (self.table._consumer_queue.empty())

        # sanity check: expect None
        assert (self.table.get_producer_index(buffer_size=-1) is None)
        assert (self.table.get_producer_index(buffer_size=0) is None)
        assert (self.table._producer_queue.qsize() == cur_p_queue_size)

        assert (self.table.get_consumer_index(buffer_size=-1) is None)
        assert (self.table.get_consumer_index(buffer_size=0) is None)
        assert (self.table._consumer_queue.qsize() == cur_c_queue_size)
        
        # sanity check: p -> c
        assert (self.table._producer_queue.qsize() == cur_p_queue_size)
        
        pidx = []
        while self.table._producer_queue.qsize() > 0:
            pidx.extend(self.table.get_producer_index(1))
        assert(len(pidx) == cur_p_queue_size)
        self.table.free_consumer_index(indices=pidx)
        assert self.table._producer_queue.qsize() == cur_p_queue_size

        pidx = self.table.get_producer_index(buffer_size=cur_p_queue_size+1)
        assert (isinstance(pidx, List) and len(pidx) == cur_p_queue_size)
        self.table.free_producer_index(pidx)
        _size_check()

        assert self.table._producer_queue.qsize() == 0, pidx
        assert (self.table._consumer_queue.qsize() == self.table.capacity)
        
        cur_p_queue_size = 0
        cur_c_queue_size = self.table.capacity

        # sanity check: c -> p
        cidx = self.table.get_consumer_index(buffer_size=cur_c_queue_size+1)
        assert (isinstance(cidx, List) and len(cidx) == cur_c_queue_size)
        self.table.free_consumer_index(cidx)
        _size_check()
        del cidx

        assert self.table._consumer_queue.qsize() == 0
        assert (self.table._producer_queue.qsize() == self.table.capacity)

    def test_to_csv(self):
        def _split_csv_line(line: str):
            line = line.replace("\n", "")
            line = line.replace(" ", "")
            return line.split("/")

        fp = "/tmp/_csvfile_dumped"
        os.makedirs(fp, exist_ok=True)
        self.table.to_csv(fp)

        buffer = self.table.buffer
        # check existance of dumped csv files
        for aid in buffer.keys():
            csv_path = os.path.join(fp, str(aid))
            assert (os.path.exists(csv_path))
            data = buffer[aid]
            col_names = [str(cname) for cname in data.keys()]
            with open(csv_path, "r") as csv:
                # check columns
                col_line = csv.readline()
                cols = _split_csv_line(col_line)
                assert (col_names == cols)
                
                for idx in range(len(next(iter(data.values())))):
                    line = csv.readline()
                    split_line = _split_csv_line(line)
                    cnames = list(data.keys())
                    assert (len(split_line) == len(cnames))
                    for cname, entry in zip(cnames, split_line):
                        assert np.all(np.abs(data[cname][idx] - np.array(eval(entry))) < 1e-5)
            os.remove(csv_path)
        os.rmdir(fp)

    def test_serialization(self):
        multi_agent = self.table.is_multi_agent
        sample_start_size = self.table._sample_start_size
        data_shapes = self.table._data_shapes
        fragment_length =  self.table._fragment_length
        data = self.table._buffer

        direct_comparable_attrs = [
            "multi_agent",
            "sample_start_size",
            "fragment_length",
        ]
        
        fp = "/tmp"
        name = "_table_dump"
        self.table.dump(fp=fp, name=name)

        path = os.path.join(fp, name+".tpkl")
        assert (os.path.exists(path))
        with open(path, "rb") as f:
            serial_dict = pkl.load(f)
        assert isinstance(serial_dict, Dict)

        for attr_name in direct_comparable_attrs:
            assert(eval(attr_name) == serial_dict.get(attr_name))

        for aid, shapes in serial_dict.get("data_shapes").items():
            assert (aid in data_shapes)
            inmem_shapes = data_shapes[aid]
            assert (shapes == inmem_shapes)
        dumped_buffer = serial_dict.get("data")
        assert (data.capacity == dumped_buffer.capacity)
        assert (data.keys() == dumped_buffer.keys())
        for k in data.keys():
            for col in data[k].keys():
                assert np.all((data[k][col] == dumped_buffer[k][col]))
        os.remove(path)

    def test_deserialization(self):
        fp = "/tmp"
        name = "_table_dump"
        self.table.dump(fp=fp, name=name)
        path = os.path.join(fp, name+".tpkl")
        table = Table.load(path)

        assert (table.is_multi_agent == self.table.is_multi_agent)
        assert (table._sample_start_size == self.table._sample_start_size)

        for aid, shapes in self.table._data_shapes.items():
            assert (aid in table._data_shapes.keys())
            assert (shapes == table._data_shapes[aid])

        
        assert (table.buffer.capacity == self.table.buffer.capacity)
        assert (table.buffer.keys() == self.table.buffer.keys())
        for k in self.table.buffer.keys():
            for col in self.table.buffer[k].keys():
                assert np.all((table.buffer[k][col] == self.table.buffer[k][col]))
        os.remove(path)

    def test_fix_table(self):
        assert (not self.table.is_fixed)

        # expect shutdown() only invoked once
        for _ in range(10):
            self.table.fix_table()

        assert (self.table.is_fixed)


@pytest.mark.parametrize(
    "dataset_config,data_shapes,dtypes,exp_cfg",
    [
        (
            {
                "episode_capacity": 1000,
                "fragment_length": 10,
                "learning_start": 0,
            },
            {
                "agent1": {EpisodeKey.CUR_OBS: (3, 4), EpisodeKey.DONE: ()},
                "agent2": {EpisodeKey.CUR_OBS: (2, 2), EpisodeKey.DONE: ()},
            },
            {EpisodeKey.CUR_OBS: np.float16, EpisodeKey.DONE: np.float16},
            {},
        ),
    ],
)
class TestOfflineDataset:
    @pytest.fixture(autouse=True)
    def _init(self, dataset_config, data_shapes, dtypes, exp_cfg):
        if not ray.is_initialized():
            ray.init(local_mode=True)

        self.dataset_config = dataset_config
        self.data_shapes = data_shapes
        self.dtypes = {k: dtypes for k in self.data_shapes}
        self.server = OfflineDataset.remote(dataset_config, exp_cfg, test_mode=False)
        self.preset_buffer_desc = BufferDescription(
            env_id="fake",
            agent_id=list(data_shapes.keys()),
            policy_id="default",
            data_shapes=data_shapes,
        )
        self.preset_table_name = Table.gen_table_name(
            env_id="fake",
            main_id=list(data_shapes.keys()),
            pid="default",
        )
        ray.get(self.server.create_table.remote(self.preset_buffer_desc))

    def _gen_data(self, size):
        data = {}
        for main_key, shape_dict in self.data_shapes.items():
            _data = {}
            for k, shape in shape_dict.items():
                full_shape = (size,) + shape
                _data[k] = (
                    np.arange(reduce(operator.mul, full_shape))
                    .reshape(full_shape)
                    .astype(self.dtypes[main_key][k])
                )
            data[main_key] = _data
        return data

    def test_save_and_sample(self):
        sizes = [10, 20, 10]
        self.preset_buffer_desc.data = [self._gen_data(v) for v in sizes]
        self.preset_buffer_desc.batch_size = sum(sizes)
        self.preset_buffer_desc.indices = ray.get(
            self.server.get_producer_index.remote(self.preset_buffer_desc)
        ).data
        ray.get(self.server.save.remote(self.preset_buffer_desc))
        data = ray.get(self.server.sample.remote(self.preset_buffer_desc))

    def test_rw_efficiency(self):
        pass

    def test_lock(self):
        non_existed_desc = {
            "agent0": BufferDescription("hello", ["agent0", "agent1"], policy_id="plain"),
            "agent1": BufferDescription("hello", ["agent0", "agent1"], policy_id="plain")
        }

        assert ray.get(self.server.lock.remote("try_lock", non_existed_desc)) == Status.FAILED
        assert ray.get(self.server.unlock.remote("try_lock", non_existed_desc)) == Status.FAILED

    def test_to_csv(self):
        sizes = [10, 20, 10]
        self.preset_buffer_desc.data = [self._gen_data(v) for v in sizes]
        self.preset_buffer_desc.batch_size = sum(sizes)
        self.preset_buffer_desc.indices = ray.get(
            self.server.get_producer_index.remote(self.preset_buffer_desc)
        ).data
        ray.get(self.server.save.remote(self.preset_buffer_desc))

        for as_csv in [True, False]:
            fp = "/tmp/_dataset_dumped"
            # content consistent ensured in Table unit test
            status = ray.get(self.server.dump.remote(fp, as_csv=as_csv))
            assert (status != Status.FAILED)
            dumped_dir = os.path.join(fp, self.preset_table_name + ("" if as_csv else ".tpkl"))
            assert (os.path.exists(dumped_dir)), as_csv
            if as_csv:
                for k in self.data_shapes.keys():
                    file_path = os.path.join(dumped_dir, str(k))
                    assert (os.path.exists(file_path))
                    os.remove(file_path)
            if as_csv:
                os.rmdir(dumped_dir)
            else:
                os.remove(dumped_dir)
            os.rmdir(fp)

    def test_load(self):
        sizes = [10, 20, 10]
        self.preset_buffer_desc.data = [self._gen_data(v) for v in sizes]
        self.preset_buffer_desc.batch_size = sum(sizes)
        self.preset_buffer_desc.indices = ray.get(
            self.server.get_producer_index.remote(self.preset_buffer_desc)
        ).data
        ray.get(self.server.save.remote(self.preset_buffer_desc))

        fp = "/tmp/_dataset_test_load"
        status = ray.get(self.server.dump.remote(fp, as_csv=False))
        ray.get(self.server.load.remote(fp))
        dumped_dir = os.path.join(fp, self.preset_table_name + ".tpkl")
        os.remove(dumped_dir)
        os.rmdir(fp)

    def test_shutdown(self):
        ray.get(self.server.shutdown.remote())

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
    
