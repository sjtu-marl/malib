from typing import Any, Dict
from abc import ABC

import copy
import numpy as np
import torch

from gym import spaces
from readerwriterlock import rwlock
from malib.utils.data import numpy_to_torch_dtype_dict


class BaseFeature(ABC):
    def __init__(
        self,
        spaces: Dict[str, spaces.Space],
        np_memory: Dict[str, np.ndarray],
        block_size: int = None,
        device: str = "cpu",
    ) -> None:
        """Constructing a feature handler for data preprocessing.

        Args:
            spaces (Dict[str, spaces.Space]): A dict of spaces
            np_memory (Dict[str, np.ndarray]): A dict of memory placeholders
            block_size (int, optional): Block size. Defaults to None.
            device (str, optional): Device name. Defaults to "cpu".
        """

        self.rw_lock = rwlock.RWLockFair()
        self._device = device
        self._spaces = spaces
        self._block_size = min(
            block_size or np.iinfo(np.longlong).max,
            list(np_memory.values())[0].shape[0],
        )
        self._available_size = 0
        self._flag = 0
        self._shared_memory = {
            k: torch.from_numpy(v[: self._block_size]).to(device).share_memory_()
            for k, v in np_memory.items()
        }

    def get(self, index: int):
        """Get data from this feature.

        Args:
            index (int): Index of the data.

        Returns:
            Any: Data
        """
        data = {}
        for k, v in self._shared_memory.items():
            data[k] = v[index]
        return data

    def write(self, data: Dict[str, Any], start: int, end: int):
        for k, v in data.items():
            # FIXME(ming): should check the size of v
            tensor = torch.as_tensor(v).to(
                self._device, dtype=self._shared_memory[k].dtype
            )
            split = 0
            if end > self.block_size:
                # we now should split the data
                split = self.block_size - start
                self._shared_memory[k][start:] = tensor[:split]
                _start = 0
                _end = tensor.shape[0] - split
            else:
                _start = start
                _end = end

            self._shared_memory[k][_start:_end] = tensor[split:]

    def generate_timestep(self) -> Dict[str, np.ndarray]:
        return {k: space.sample() for k, space in self.spaces.items()}

    def generate_batch(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        batch = {}
        for k, space in self.spaces.items():
            data = np.stack(
                [space.sample() for _ in range(batch_size)], dtype=space.dtype
            )
            batch[k] = data
        return batch

    @property
    def spaces(self) -> Dict[str, spaces.Space]:
        return copy.deepcopy(self._spaces)

    @property
    def block_size(self) -> int:
        return self._block_size

    def __len__(self):
        return self._available_size

    def safe_get(self, index: int):
        with self.rw_lock.gen_rlock():
            if len(self) == 0:
                raise IndexError(f"index:{index} exceeds for available size is 0")
            elif index >= len(self):
                # re-sampling
                index = index % self._available_size
            return self.get(index)

    def safe_put(self, data: Any, batch_size: int):
        with self.rw_lock.gen_wlock():
            # request segment asscessment
            self.write(data, self._flag, self._flag + batch_size)
            self._flag = (self._flag + batch_size) % self._block_size
            self._available_size = min(
                self._available_size + batch_size, self._block_size
            )
