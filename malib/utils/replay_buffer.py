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

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check, Sequence
from copy import deepcopy
from collections import defaultdict

import h5py
import torch
import pickle
import numpy as np

from malib.utils.tianshou_batch import (
    Batch,
    _alloc_by_keys_diff,
    _create_value,
    _parse_value,
)


@no_type_check
def to_numpy(x: Any) -> Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    else:  # fallback
        return np.asanyarray(x)


# Note: object is used as a proxy for objects that can be pickled
# Note: mypy does not support cyclic definition currently
Hdf5ConvertibleValues = Union[  # type: ignore
    int,
    float,
    Batch,
    np.ndarray,
    torch.Tensor,
    object,
    "Hdf5ConvertibleType",  # type: ignore
]

Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]  # type: ignore


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs
    ) -> None:
        self.capacity = size
        self.data = {}
        self.flag = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_batch(self, data: Dict[str, np.ndarray]):
        any_v = list(data.values())[0]
        shift = min(0, self.capacity - any_v.shape[0] - self.flag)

        for k, v in data.items():
            if k not in self.data:
                self.data[k] = np.zeros((self.capacity,) + v.shape[1:], dtype=v.dtype)
            assert v.shape[0] == any_v.shape[0], (any_v.shape, v.shape, k)
            self.data[k] = np.roll(self.data[k], shift=shift, axis=0)
            self.data[k][self.flag + shift : self.flag + shift + v.shape[0]] = v

        self.flag = (self.flag + shift + any_v.shape[0]) % self.capacity
        self.size = min(self.size + any_v.shape[0], self.capacity)

    def sample_indices(self, batch_size: int) -> Sequence[int]:
        indices = np.random.choice(self.size, batch_size)
        return indices

    def sample(self, batch_size: int) -> Tuple[Batch, List[int]]:
        indices = self.sample_indices(batch_size)
        samples = {k: v[indices] for k, v in self.data.items()}
        return Batch(samples), indices


class MultiagentReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            size, stack_num, ignore_obs_next, save_only_last_obs, sample_avail, **kwargs
        )

        self.data = defaultdict(
            lambda: ReplayBuffer(
                size=size,
                stack_num=stack_num,
                ignore_obs_next=ignore_obs_next,
                save_only_last_obs=save_only_last_obs,
                sample_avail=sample_avail,
                **kwargs
            )
        )

    def add_batch(self, data: Dict[str, Dict[str, np.ndarray]]):
        for agent, _data in data.items():
            self.data[agent].add_batch(_data)

        size_candidates = set()
        for e in self.data.values():
            size_candidates.add(e.size)
        assert len(size_candidates) == 1, (
            size_candidates,
            {k: v.size for k, v in self.data.items()},
        )
        self.size = list(self.data.values())[0].size

    def sample(self, batch_size: int) -> Dict[str, Tuple[Batch, List[int]]]:
        agent_batch_tups = {
            agent: data.sample(batch_size) for agent, data in self.data.items()
        }

        return agent_batch_tups
