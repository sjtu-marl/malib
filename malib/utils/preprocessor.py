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

from abc import ABCMeta, abstractmethod
from typing import Dict, Sequence, Tuple, List, Any
from functools import reduce

import operator
import numpy as np

from gym import spaces

from malib.utils.typing import DataTransferType


def _get_batched(data: Any):
    """Get batch dim, nested data must be numpy array like"""

    res = []
    if isinstance(data, Dict):
        for k, v in data.items():
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i >= len(res):
                    res.append({})
                res[i][k] = e
    elif isinstance(data, Sequence):
        for v in data:
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i >= len(res):
                    res.append([])
                res[i].append(e)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unexpected nested data type: {type(data)}")

    return res


class Preprocessor(metaclass=ABCMeta):
    def __init__(self, space: spaces.Space):
        self._original_space = space
        self._size = 0

    @property
    def original_space(self) -> spaces.Space:
        return self._original_space

    @abstractmethod
    def transform(self, data, nested=False) -> DataTransferType:
        """Transform original data to feet the preprocessed shape. Nested works for nested array."""
        pass

    @abstractmethod
    def write(self, array: DataTransferType, offset: int, data: Any):
        pass

    @property
    def size(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            self.shape,
            dtype=np.float32,
        )


class DictFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Dict):
        assert isinstance(space, spaces.Dict), space
        super(DictFlattenPreprocessor, self).__init__(space)
        self._preprocessors = {}

        for k, _space in space.spaces.items():
            self._preprocessors[k] = get_preprocessor(_space)(_space)

        self._size = sum([prep.size for prep in self._preprocessors.values()])

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return self._size

    def transform(self, data, nested=False) -> DataTransferType:
        """Transform support multi-instance input"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, Dict):
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        elif isinstance(data, (list, tuple)):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, dict):
            for k, _data in sorted(data.items()):
                size = self._preprocessors[k].size
                array[offset : offset + size] = self._preprocessors[k].transform(_data)
                offset += size
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class TupleFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Tuple):
        """Init a tuple flatten preprocessor, will stack inner flattend spaces.

        Note:
            All sub spaces in a tuple should be homogeneous.

        Args:
            space (spaces.Tuple): A tuple of homogeneous spaces.
        """
        super(TupleFlattenPreprocessor, self).__init__(space)
        self._preprocessors = []

        self._preprocessors.append(get_preprocessor(space.spaces[0])(space.spaces[0]))
        expected_shape = self._preprocessors[0].shape
        self._size += self._preprocessors[0].size

        for _space in space.spaces[1:]:
            sub_preprocessor = get_preprocessor(_space)(_space)
            assert sub_preprocessor.shape == expected_shape, (
                sub_preprocessor.shape,
                expected_shape,
            )
            self._preprocessors.append(sub_preprocessor)
            self._size += sub_preprocessor.size
        self._shape = (len(space.spaces),) + expected_shape

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    def transform(self, data, nested=False) -> DataTransferType:
        if nested:
            raise NotImplementedError

        if isinstance(data, List):
            # write batch
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            # write single to stack
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, Tuple):
            for _data, prep in zip(data, self._preprocessors):
                array[offset : offset + 1] = prep.transform(_data)
                offset += 1
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class BoxFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxFlattenPreprocessor, self).__init__(space)
        self._size = reduce(operator.mul, space.shape)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        if nested:
            data = _get_batched(data)

        if isinstance(data, list):
            array = np.vstack(data)
            return array
        else:
            array = np.asarray(data).reshape(self.shape)
        return array

    def write(self, array, offset, data):
        pass


class BoxStackedPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxStackedPreprocessor, self).__init__(space)
        assert (
            len(space.shape) >= 3
        ), "Stacked box preprocess can only applied to 3D shape"
        self._size = reduce(operator.mul, space.shape)
        self._shape = space.shape

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    def transform(self, data, nested=False) -> DataTransferType:
        if nested:
            raise TypeError("Do not support nested transformation yet")

        if isinstance(data, list):
            array = np.stack(data)
            return array
        else:
            array = np.asarray(data)
            return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        pass


class DiscreteFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Discrete):
        super(DiscreteFlattenPreprocessor, self).__init__(space)
        self._size = space.n

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        """Transform to one hot"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, int):
            array = np.zeros(self.size, dtype=np.float32)
            array[data] = 1
            return array
        elif isinstance(data, np.ndarray):
            array = data.reshape((-1, self.size))
            return array
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

    def write(self, array, offset, data):
        pass


class Mode:
    FLATTEN = "flatten"
    STACK = "stack"


def get_preprocessor(space: spaces.Space, mode: str = Mode.FLATTEN):
    if mode == Mode.FLATTEN:
        if isinstance(space, spaces.Dict):
            # logger.debug("Use DictFlattenPreprocessor")
            return DictFlattenPreprocessor
        elif isinstance(space, spaces.Tuple):
            # logger.debug("Use TupleFlattenPreprocessor")
            return TupleFlattenPreprocessor
        elif isinstance(space, spaces.Box):
            # logger.debug("Use BoxFlattenPreprocessor")
            return BoxFlattenPreprocessor
        elif isinstance(space, spaces.Discrete):
            return DiscreteFlattenPreprocessor
        else:
            raise TypeError(f"Unexpected space type: {type(space)}")
    elif mode == Mode.STACK:  # for sequential model like CNN and RNN
        if isinstance(space, spaces.Box):
            return BoxStackedPreprocessor
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"Unexpected mode: {mode}")
