import abc
import math
import sys
from typing import Union, Iterable

import numpy as np

from malib import settings
from malib.utils.typing import DataTransferType


class DataArray(abc.ABC):
    def __init__(
        self,
        name: str,
        capacity: Union[int, None] = settings.DEFAULT_EPISODE_CAPACITY,
        init_capacity: int = settings.DEFAULT_EPISODE_INIT_CAPACITY,
    ):
        self._name = name
        self._length = 0

        if capacity == -1:
            self._max_capacity = sys.maxsize
        elif capacity is None:
            self._max_capacity = settings.DEFAULT_EPISODE_CAPACITY
        else:
            self._max_capacity = capacity

        self._capacity = min(init_capacity, capacity)
        self._offset = 0
        self._data = None

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def capacity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fill(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def insert(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError


class NumpyDataArray(DataArray):
    def __init__(
        self,
        name: str,
        capacity: Union[int, None] = settings.DEFAULT_EPISODE_CAPACITY,
        init_capacity: int = settings.DEFAULT_EPISODE_INIT_CAPACITY,
    ):
        """Data storage in the form of numpy array. The instantiation of its physical storage space is
        done when the first data insert/fill operation is called. Doubling its current capacity
        until hit the maximum capacity limitation when an insertion is called on a full data array.

        :param str name: A parameter help users identifying the data array, will set to be consistent with the episode
                                column names by default.
        :param int capacity: The max capacity that the data array can span. The default value
                            malib.settings.DEFAULT_EPISODE_CAPACITY is applied when ***capacity*** is set to be None.
        :param int init_capacity: Then init size of the internal data array created when the first insert/fill operation
                        is called.
        """
        super().__init__(name=name, capacity=capacity, init_capacity=init_capacity)

    def __getitem__(self, item: Union[int, Iterable, slice]) -> np.ndarray:
        """
        :param Union[int,Iterable,slice] item: indices of the desired data entry.

        :return: data in the form of numpy ndarray.

        :raises IndexError
        """
        if self._data is None:
            raise IndexError("Index on column data out of range")

        if isinstance(item, int) or isinstance(item, np.integer):
            idx = item
            # check valid index
            if -self._length <= idx < self._length:
                if idx >= 0:
                    idx = (self._offset + idx) % self._max_capacity
                else:
                    idx = (self._offset + self._length + idx) % self._max_capacity
                return self._data[idx]
            else:
                raise IndexError(
                    f"Index on column data out of range, length{self._length}, index{idx}"
                )
        elif isinstance(item, Iterable):
            indices = np.asarray(item)
            indices = (
                (indices < 0) * self._length + indices + self._offset
            ) % self._max_capacity
            return self._data[indices]
        elif isinstance(item, slice):
            return self[range(*item.indices(self._length))]
        else:
            raise IndexError(
                f"Indices must be int, numpy.integer, slices, not {item.__class__}"
            )

    def __len__(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._length

    def fill(self, data: DataTransferType, capacity: int = None) -> "NumpyDataArray":
        """
        Flush fill the array with the input data.

        Note: For performance issue, it is designed that after the filling, the internal data storage will only use a
            shallow copy of the input data. If users are not sure if the input data will be modified afterwards, please
            use the deep copy of the data instead.

        :param DataTransferType data: Array-like data input.
        :param int capacity: Override the pre-set max capacity if provided.
        :return: A reference of the caller.
        """
        if capacity is not None:
            self._max_capacity = capacity
            if data.shape[0] >= capacity:
                data = data[-capacity:]
            self._data = data
            self._capacity = data.shape[0]
        else:
            self._data = data
            self._capacity = data.shape[0]

        self._offset = 0
        self._length = data.shape[0]
        return self

    def insert(self, data: DataTransferType) -> None:
        """
        Insert data at the back of the array. A certain extent of time consumptions are expected due to:
        (1) The first insert/fill operation will instantiate the physical storage space.
        (2) Possible memory allocation & data movement when insert on a full array.

        :param DataTransferType data: Data block to be inserted at the back of the array.
        :return: None
        """

        if self._data is None:
            data_shape = list(data.shape)
            data_shape[0] = max(self._capacity, data.shape[0])
            self._data = np.empty_like(data, shape=data_shape)

        length = data.shape[0]
        assert 0 <= self._length <= self._capacity <= self._max_capacity
        if self._length == self._max_capacity:
            if length < self._capacity:
                indices = (self._offset + np.array(range(length))) % self._capacity
                self._data[indices] = data
                self._offset = (indices[-1] + 1) % self._capacity
            else:
                self._data = data[-self._capacity :]
                self._offset = 0
        else:
            # print(self._capacity)
            assert self._offset == 0
            target_length = self._length + length
            if target_length <= self._capacity:
                self._data[self._length : target_length] = data
                self._length = target_length
            else:
                new_capacity = min(2 * target_length, self._max_capacity)
                data_shape = list(self._data.shape)
                data_shape[0] = new_capacity
                _data = np.empty_like(self._data, shape=data_shape)
                _data[: self._length] = self._data[: self._length]
                inserted_length = min(new_capacity - self._length, length)
                _data[self._length : self._length + inserted_length] = data[
                    :inserted_length
                ]
                data = data[inserted_length:]
                self._data = _data
                self._length += inserted_length
                self._capacity = new_capacity
                if data.size > 0:
                    self.insert(data)

    def get_data(self) -> DataTransferType:
        """
        Return the existing data in the internal storage, order preserving.

        :return DataTransferType
        """
        # FIXME(ming):
        indices = np.roll(np.arange(self._length), self._offset)
        return self._data[indices]

    @property
    def nbytes(self) -> int:
        """
        Return the memory size(in bytes) of the space occupied by the array data.

        :return: int
        """

        return self._data.nbytes


class LinkDataArray(DataArray):
    """
    Segmented data structure for efficient rollout data storage/inference
    """

    from collections import deque

    def __init__(
        self,
        name: str,
        capacity: Union[int, None] = settings.DEFAULT_EPISODE_CAPACITY,
        init_capacity: Union[int, None] = settings.DEFAULT_EPISODE_INIT_CAPACITY,
        block_size: Union[int, None] = settings.DEFAULT_CONFIG["rollout"][
            "fragment_length"
        ],
    ):
        self._max_blocks = math.ceil(capacity / block_size)

        super().__init__(
            name=name,
            capacity=self._max_blocks * block_size,
            init_capacity=self._max_blocks * block_size,
        )

        self._blocks = self.deque(maxlen=self._max_blocks)
        self._block_lengths = 0
        self._length = 0
        self._block_size = 75

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __getitem__(self, item: int) -> np.ndarray:
        idx = item
        if isinstance(item, int) or isinstance(item, np.integer):
            # check valid index
            if -self._length <= idx < self._length:
                block_index = item / self._block_size
                inblock_index = item % self._block_size
                return self._blocks[math.floor(block_index)][inblock_index]
            else:
                raise IndexError("Index on column data out of range")
        elif isinstance(item, Iterable):
            indices = np.array(item)
            block_indices = np.floor_divide(indices, self._block_size)
            inblock_indices = np.mod(indices, self._block_size)
            res = []
            for (x, y) in zip(block_indices, inblock_indices):
                res.append(self._blocks[x][y])
            return np.array(res)
        elif isinstance(item, slice):
            return self[range(*item.indices(self._length))]
        else:
            raise IndexError(
                f"Indices must be int, numpy.integer, slices, not {item.__class__}"
            )

    def __len__(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._length

    def fill(
        self, data: np.ndarray, capacity: Union[None, int] = None
    ) -> "LinkDataArray":
        if capacity is not None:
            if capacity % self._block_size:
                self._max_blocks = math.ceil(capacity / self._block_size)
                self._capacity = self._max_blocks * self._block_size
                self._blocks = self.deque(maxlen=self._max_blocks)
        else:
            self._blocks = self.deque(maxlen=self._blocks.maxlen)

        if self._capacity > data.shape[0]:
            data = data[-self._capacity :]

        while data.size > 0:
            self._blocks.append(data[: self._block_size])
            data = data[self._block_size :]
        self._length = self._block_size * len(self._blocks)
        return self

    def insert(self, data: Union[np.ndarray, Iterable[np.ndarray]]):
        if isinstance(data, np.ndarray):
            assert data.shape[0] % self._block_size == 0
            while data.size > 0:
                self._blocks.append(data[: self._block_size])
                data = data[self._block_size :]
        elif isinstance(data, self.deque):
            for block in data:
                assert isinstance(block, np.ndarray)
                assert block.shape[0] == self._block_size

                self._blocks.append(block)

        self._length = self._block_size * len(self._blocks)

    def get_data(self) -> Iterable[np.ndarray]:
        return self._blocks
