# refer to: https://github.com/thu-ml/tianshou/blob/master/tianshou/data/buffer/base.py

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check
from copy import deepcopy

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


def to_hdf5(
    x: Hdf5ConvertibleType, y: h5py.Group, compression: Optional[str] = None
) -> None:
    """Copy object into HDF5 group."""

    def to_hdf5_via_pickle(
        x: object, y: h5py.Group, key: str, compression: Optional[str] = None
    ) -> None:
        """Pickle, convert to numpy array and write to HDF5 dataset."""
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data, compression=compression)

    for k, v in x.items():
        if isinstance(v, (Batch, dict)):
            # dicts and batches are both represented by groups
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
                subgrp.attrs["__data_type__"] = "Batch"
            else:
                subgrp_data = v
            to_hdf5(subgrp_data, subgrp, compression=compression)
        elif isinstance(v, torch.Tensor):
            # PyTorch tensors are written to datasets
            y.create_dataset(k, data=to_numpy(v), compression=compression)
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                # NumPy arrays are written to datasets
                y.create_dataset(k, data=v, compression=compression)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k, compression=compression)
                except Exception as exception:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed."
                    ) from exception
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            # ints and floats are stored as attributes of groups
            y.attrs[k] = v
        else:  # resort to pickle for any other type of object
            try:
                to_hdf5_via_pickle(v, y, k, compression=compression)
            except Exception as exception:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed."
                ) from exception
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(x: h5py.Group, device: Optional[str] = None) -> Hdf5ConvertibleValues:
    """Restore object from HDF5 group."""
    if isinstance(x, h5py.Dataset):
        # handle datasets
        if x.attrs["__data_type__"] == "ndarray":
            return np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            return torch.tensor(x, device=device)
        else:
            return pickle.loads(x[()])
    else:
        # handle groups representing a dict or a Batch
        y = dict(x.attrs.items())
        data_type = y.pop("__data_type__", None)
        for k, v in x.items():
            y[k] = from_hdf5(v, device)
        return Batch(y) if data_type == "Batch" else y


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction \
    between the policy and environment.

    ReplayBuffer can be considered as a specialized form (or management) of Batch. It
    stores all the data in a batch with circular-queue style.

    For the example usage of ReplayBuffer, please check out Section Buffer in
    :doc:`/tutorials/concepts`.

    :param int size: the maximum size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater than or
        equal to 1. Default to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next. Default to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has a shape
        of (timestep, ...) because of temporal stacking. Default to False.
    :param bool sample_avail: the parameter indicating sampling only available index
        when using frame-stack sampling method. Default to False.
    """

    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
    ) -> None:
        self.options: Dict[str, Any] = {
            "stack_num": stack_num,
            "ignore_obs_next": ignore_obs_next,
            "save_only_last_obs": save_only_last_obs,
            "sample_avail": sample_avail,
        }
        super().__init__()
        self.maxsize = int(size)
        assert stack_num > 0, "stack_num should be greater than 0"
        self.stack_num = stack_num
        self._indices = np.arange(size)
        self._save_obs_next = not ignore_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._sample_avail = sample_avail
        self._meta: Batch = Batch()
        self._ep_rew: Union[float, np.ndarray]
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        try:
            return self._meta[key]
        except KeyError as exception:
            raise AttributeError from exception

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        We need it because pickling buffer does not work out-of-the-box
        ("buffer.__getattr__" is customized).
        """
        self.__dict__.update(state)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        assert (
            key not in self._reserved_keys
        ), "key '{}' is reserved and cannot be assigned".format(key)
        super().__setattr__(key, value)

    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__dict__, f, compression=compression)

    @classmethod
    def load_hdf5(cls, path: str, device: Optional[str] = None) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))  # type: ignore
        return buf

    @classmethod
    def from_data(
        cls,
        obs: h5py.Dataset,
        act: h5py.Dataset,
        rew: h5py.Dataset,
        done: h5py.Dataset,
        obs_next: h5py.Dataset,
    ) -> "ReplayBuffer":
        size = len(obs)
        assert all(
            len(dset) == size for dset in [obs, act, rew, done, obs_next]
        ), "Lengths of all hdf5 datasets need to be equal."
        buf = cls(size)
        if size == 0:
            return buf
        batch = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
        buf.set_batch(batch)
        buf._size = size
        return buf

    def reset(self, keep_statistics: bool = False) -> None:
        """Clear all the data in replay buffer and episode statistics."""
        self.last_index = np.array([0])
        self._index = self._size = 0
        if not keep_statistics:
            self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, 0

    def set_batch(self, batch: Batch) -> None:
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize and set(batch.keys()).issubset(
            self._reserved_keys
        ), "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        last = (self._index - 1) % self._size if self._size else 0
        return np.array([last] if not self.done[last] and self._size else [], int)

    def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of an episode.
        """
        index = (index - 1) % self._size
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + end_flag) % self._size

    def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of an episode.
        """
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], int)
        stack_num, buffer.stack_num = buffer.stack_num, 1
        from_indices = buffer.sample_indices(0)  # get all available indices
        buffer.stack_num = stack_num
        if len(from_indices) == 0:
            return np.array([], int)
        to_indices = []
        for _ in range(len(from_indices)):
            to_indices.append(self._index)
            self.last_index[0] = self._index
            self._index = (self._index + 1) % self.maxsize
            self._size = min(self._size + 1, self.maxsize)
        to_indices = np.array(to_indices)
        if self._meta.is_empty():
            self._meta = _create_value(  # type: ignore
                buffer._meta, self.maxsize, stack=False
            )
        self._meta[to_indices] = buffer._meta[from_indices]
        return to_indices

    def _add_index(
        self, rew: Union[float, np.ndarray], done: bool
    ) -> Tuple[int, Union[float, np.ndarray], int, int]:
        """Maintain the buffer's state after adding one data batch.

        Return (index_to_be_modified, episode_reward, episode_length,
        episode_start_index).
        """
        self.last_index[0] = ptr = self._index
        self._size = min(self._size + 1, self.maxsize)
        self._index = (self._index + 1) % self.maxsize

        self._ep_rew += rew
        self._ep_len += 1

        if done:
            result = ptr, self._ep_rew, self._ep_len, self._ep_idx
            self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, self._index
            return result
        else:
            return ptr, self._ep_rew * 0.0, 0, self._ep_idx

    def add_episode(self, episode: Dict[str, np.ndarray]):
        assert set(["obs", "act", "rew", "done"]).issubset(
            episode.keys()
        ), episode.keys()

        length = len(episode["obs"])
        for i in range(length):
            self.add({k: v[i] for k, v in episode.items()})

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into replay buffer.

        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.
        :param buffer_ids: to make consistent with other buffer's add function; if it
            is not None, we assume the input batch's first dimension is always 1.

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys()), batch.keys()
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (
                batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
            )
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = list(
            map(lambda x: np.array([x]), self._add_index(rew, done))
        )
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(batch, self.maxsize, stack)  # type: ignore
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._index, self._size), np.arange(self._index)]
                )
            else:
                return np.array([], int)
        else:
            if batch_size < 0:
                return np.array([], int)
            all_indices = prev_indices = np.concatenate(
                [np.arange(self._index, self._size), np.arange(self._index)]
            )
            for _ in range(self.stack_num - 2):
                prev_indices = self.prev(prev_indices)
            all_indices = all_indices[prev_indices != self.prev(prev_indices)]
            if batch_size > 0:
                return np.random.choice(all_indices, batch_size)
            else:
                return all_indices

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        return self[indices], indices

    def get(
        self,
        index: Union[int, List[int], np.ndarray],
        key: str,
        default_value: Any = None,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g., if you set ``key = "obs", stack_num = 4, index = t``, it returns the
        stacked result as ``[obs[t-3], obs[t-2], obs[t-1], obs[t]]``.

        :param index: the index for getting stacked data.
        :param str key: the key to get, should be one of the reserved_keys.
        :param default_value: if the given key's data is not found and default_value is
            set, return this default_value.
        :param int stack_num: Default to self.stack_num.
        """
        if key not in self._meta and default_value is not None:
            return default_value
        val = self._meta[key]
        if stack_num is None:
            stack_num = self.stack_num
        try:
            if stack_num == 1:  # the most often case
                return val[index]
            stack: List[Any] = []
            if isinstance(index, list):
                indices = np.array(index)
            else:
                indices = index  # type: ignore
            for _ in range(stack_num):
                stack = [val[indices]] + stack
                indices = self.prev(indices)
            if isinstance(val, Batch):
                return Batch.stack(stack, axis=indices.ndim)
            else:
                return np.stack(stack, axis=indices.ndim)
        except IndexError as exception:
            if not (isinstance(val, Batch) and val.is_empty()):
                raise exception  # val != Batch()
            return Batch()

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index  # type: ignore
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indices, "obs")
        if self._save_obs_next:
            obs_next = self.get(indices, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indices), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indices],
            rew=self.rew[indices],
            done=self.done[indices],
            obs_next=obs_next,
            info=self.get(indices, "info", Batch()),
            policy=self.get(indices, "policy", Batch()),
        )
