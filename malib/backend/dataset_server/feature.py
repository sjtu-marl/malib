from typing import Any, Dict
from abc import ABC, abstractmethod

import copy
import numpy as np
import torch

from gym import spaces
from readerwriterlock import rwlock


numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


class BaseFeature(ABC):
    def __init__(
        self,
        spaces: Dict[str, spaces.Space],
        np_memory: Dict[str, np.ndarray],
        block_size: int = None,
        device: str = "cpu",
    ) -> None:
        self.rw_lock = rwlock.RWLockFair()
        self._device = device
        self._spaces = spaces
        self._block_size = block_size or list(np_memory.values())[0].shape[0]
        self._available_size = 0
        self._flag = 0
        self._shared_memory = {
            k: torch.from_numpy(v).to(device).share_memory_()
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
            self._shared_memory[k][start:end] = torch.as_tensor(v).to(
                self._device, dtype=self._shared_memory[k].dtype
            )

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
