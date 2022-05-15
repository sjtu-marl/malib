# MIT License

# Copyright (c) 2021 MARL @ SJTU

# reference: https://github.com/thu-ml/tianshou/blob/master/tianshou/data/batch.py

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

from typing import Any, Union, Optional, Collection
from numbers import Number

import torch
import numpy as np


def _is_scalar(value: Any) -> bool:
    # check if the value is a scalar
    # 1. python bool object, number object: isinstance(value, Number)
    # 2. numpy scalar: isinstance(value, np.generic)
    # 3. python object rather than dict / Batch / tensor
    # the check of dict / Batch is omitted because this only checks a value.
    # a dict / Batch will eventually check their values
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.shape
    else:
        # np.asanyarray will cause dead loop in some cases
        return np.isscalar(value)


def _is_number(value: Any) -> bool:
    # isinstance(value, Number) checks 1, 1.0, np.int(1), np.float(1.0), etc.
    # isinstance(value, np.nummber) checks np.int32(1), np.float64(1.0), etc.
    # isinstance(value, np.bool_) checks np.bool_(True), etc.
    # similar to np.isscalar but np.isscalar('st') returns True
    return isinstance(value, (Number, np.number, np.bool_))


def _to_array_with_correct_type(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray) and issubclass(
        obj.dtype.type, (np.bool_, np.number)
    ):
        return obj  # most often case
    # convert the value to np.ndarray
    # convert to object obj type if neither bool nor number
    # raises an exception if array's elements are tensors themselves
    obj_array = np.asanyarray(obj)
    if not issubclass(obj_array.dtype.type, (np.bool_, np.number)):
        obj_array = obj_array.astype(object)
    if obj_array.dtype == object:
        # scalar ndarray with object obj type is very annoying
        # a=np.array([np.array({}, dtype=object), np.array({}, dtype=object)])
        # a is not array([{}, {}], dtype=object), and a[0]={} results in
        # something very strange:
        # array([{}, array({}, dtype=object)], dtype=object)
        if not obj_array.shape:
            obj_array = obj_array.item(0)
        elif all(isinstance(arr, np.ndarray) for arr in obj_array.reshape(-1)):
            return obj_array  # various length, np.array([[1], [2, 3], [4, 5, 6]])
        elif any(isinstance(arr, torch.Tensor) for arr in obj_array.reshape(-1)):
            raise ValueError("Numpy arrays of tensors are not supported yet.")
    return


def _parse_value(obj: Any) -> Optional[Union[np.ndarray, torch.Tensor]]:
    if (
        (
            isinstance(obj, np.ndarray)
            and issubclass(obj.dtype.type, (np.bool_, np.number))
        )
        or isinstance(obj, torch.Tensor)
        or obj is None
    ):  # third often case
        return obj
    elif _is_number(obj):  # second often case, but it is more time-consuming
        return np.asanyarray(obj)
    else:
        if (
            not isinstance(obj, np.ndarray)
            and isinstance(obj, Collection)
            and len(obj) > 0
            and all(isinstance(element, torch.Tensor) for element in obj)
        ):
            try:
                return torch.stack(obj)  # type: ignore
            except RuntimeError as exception:
                raise TypeError(
                    "Batch does not support non-stackable iterable"
                    " of torch.Tensor as unique value yet."
                ) from exception
        # None, scalar, normal obj list (main case)
        # or an actual list of objects
        try:
            obj = _to_array_with_correct_type(obj)
        except ValueError as exception:
            raise TypeError(
                "Batch does not support heterogeneous list/"
                "tuple of tensors as unique value yet."
            ) from exception
        return obj


def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> torch.Tensor:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


def hard_update(target, source):
    for t_para, s_para in zip(target.parameters(), source.parameters()):
        t_para.data.copy_(s_para.data)


def soft_update(target, source, rho):
    for t_para, s_para in zip(target.parameters(), source.parameters()):
        t_para.data.copy_(t_para.data * rho + s_para.data * (1 - rho))
