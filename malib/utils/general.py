from typing import OrderedDict

import torch
import numpy as np
import copy

from malib import settings
from malib.utils.typing import Dict, Callable


def update_configs(update_dict, ori_dict=None):
    """Update global configs with a given dict"""

    ori_configs = (
        copy.copy(ori_dict)
        if ori_dict is not None
        else copy.copy(settings.DEFAULT_CONFIG)
    )

    for k, v in update_dict.items():
        # assert k in ori_configs, f"Illegal key: {k}, {list(ori_configs.keys())}"
        if isinstance(v, dict):
            ph = ori_configs[k] if isinstance(ori_configs.get(k), dict) else {}
            ori_configs[k] = update_configs(v, ph)
        else:
            ori_configs[k] = copy.copy(v)
    return ori_configs


# TODO(ming): will be replaced with many dicts
def iter_dicts_recursively(d1, d2):
    """Assuming dicts have the exact same structure."""
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


def iter_many_dicts_recursively(*d, history=None):
    """Assuming dicts have the exact same structure, or raise KeyError."""

    for k, v in d[0].items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iter_many_dicts_recursively(
                *[_d[k] for _d in d],
                history=history + [k] if history is not None else None,
            )
        else:
            if history is None:
                yield d, k, tuple([_d[k] for _d in d])
            else:
                yield history + [k], d, k, tuple([_d[k] for _d in d])


class BufferDict(dict):
    @property
    def capacity(self) -> int:
        capacities = []
        for _, _, v in iterate_recursively(self):
            capacities.append(v.shape[0])
        return max(capacities)

    def index(self, indices):
        return self.index_func(self, indices)

    def index_func(self, x, indices):
        if isinstance(x, (dict, BufferDict)):
            res = BufferDict()
            for k, v in x.items():
                res[k] = self.index_func(v, indices)
            return res
        else:
            t = x[indices]
            # Logger.debug("sampled data shape: {} {}".format(t.shape, indices))
            return t

    def set_data(self, index, new_data):
        return self.set_data_func(self, index, new_data)

    def set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, BufferDict)):
            for nk, nv in new_data.items():
                self.set_data_func(x[nk], index, nv)
        else:
            if isinstance(new_data, torch.Tensor):
                t = new_data.cpu().numpy()
            elif isinstance(new_data, np.ndarray):
                t = new_data
            else:
                raise TypeError(
                    f"Unexpected type for new insert data: {type(new_data)}, expected is np.ndarray"
                )
            x[index] = t.copy()


def iterate_recursively(d: Dict):
    for k, v in d.items():
        if isinstance(v, (dict, BufferDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v


def _default_dtype_mapping(dtype):
    # FIXME(ming): cast 64 to 32?
    if dtype in [np.int32, np.int64, int]:
        return torch.int32
    elif dtype in [float, np.float32, np.float16]:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype in [bool, np.bool_]:
        return torch.float32
    else:
        raise NotImplementedError(f"dtype: {dtype} has no transmission rule.") from None


# wrap with type checking
def _walk(caster, v):
    if isinstance(v, (dict, BufferDict)):
        for k, _v in v.items():
            v[k] = _walk(caster, _v)
    else:
        v = caster(v)
    return v


def tensor_cast(
    custom_caster: Callable = None,
    callback: Callable = None,
    dtype_mapping: Dict = None,
    device="cpu",
):
    """Casting the inputs of a method into tensors if needed.

    Note:
        This function does not support recursive iteration.

    Args:
        custom_caster (Callable, optional): Customized caster. Defaults to None.
        callback (Callable, optional): Callback function, accepts returns of wrapped function as inputs. Defaults to None.
        dtype_mapping (Dict, optional): Specify the data type for inputs which you wanna. Defaults to None.

    Returns:
        Callable: A decorator.
    """
    dtype_mapping = dtype_mapping or _default_dtype_mapping
    cast_to_tensor = custom_caster or (
        lambda x: torch.FloatTensor(x.copy()).to(
            device=device, dtype=dtype_mapping(x.dtype)
        )
        if not isinstance(x, torch.Tensor)
        else x
    )

    def decorator(func):
        def wrap(*args, **kwargs):
            new_args = []
            for i, arg in enumerate(args):
                new_args.append(_walk(cast_to_tensor, arg))
            for k, v in kwargs.items():
                kwargs[k] = _walk(cast_to_tensor, v)
            rets = func(*new_args, **kwargs)
            if callback is not None:
                callback(rets)
            return rets

        return wrap

    return decorator
