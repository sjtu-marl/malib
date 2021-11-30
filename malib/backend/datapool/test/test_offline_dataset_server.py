import operator
import pytest
import ray
import numpy as np

from functools import reduce

from malib.utils.typing import BufferDescription, List, Any, Dict, Tuple
from malib.utils.episode import EpisodeKey
from malib.backend.datapool.offline_dataset_server import (
    BufferDict,
    OfflineDataset,
    Table,
)


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
    "keys,capacity,fragment_length,shared_data_shapes,sample_start_size",
    [
        (
            [1, 2],
            1000,
            10,
            {EpisodeKey.CUR_OBS: (3, 4), EpisodeKey.REWARD: (), EpisodeKey.DONE: ()},
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
        sample_start_size: int,
        event_loop=None,
        mode: str = "queue",
    ):
        data_shapes = {k: shared_data_shapes for k in keys}
        data_dtypes = {k: {_k: np.float16 for _k in shared_data_shapes} for k in keys}
        self.table = Table(
            capacity,
            fragment_length,
            data_shapes,
            data_dtypes,
            sample_start_size,
            event_loop,
            mode,
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
        pass

    def test_serialization(self):
        pass

    def test_deserialization(self):
        pass


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
        self.dtypes = dtypes
        self.server = OfflineDataset.remote(dataset_config, exp_cfg, test_mode=False)
        self.preset_buffer_desc = BufferDescription(
            env_id="fake",
            agent_id=list(data_shapes.keys()),
            policy_id="default",
            data_shapes=data_shapes,
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
                    .astype(self.dtypes[k])
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
