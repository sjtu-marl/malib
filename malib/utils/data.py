# reference: https://github.com/thu-ml/tianshou/blob/master/tianshou/data/batch.py


from typing import Any, Union, Optional, Collection, Dict
from numbers import Number

import torch
import numpy as np

from numba import njit


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


def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape, dtype=np.float32)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(rew.shape[0] - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)


class Postprocessor:
    @staticmethod
    def gae_return(
        state_value,
        next_state_value,
        reward,
        done,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):

        adv = _gae_return(
            state_value, next_state_value, reward, done, gamma, gae_lambda
        )

        return adv

    @staticmethod
    def compute_episodic_return(
        batch: Dict[str, Any],
        state_value: np.ndarray = None,
        next_state_value: np.ndarray = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        if isinstance(batch["rew"], torch.Tensor):
            rew = batch["rew"].cpu().numpy()
        else:
            rew = batch["rew"]

        if isinstance(batch["done"], torch.Tensor):
            done = batch["done"].cpu().numpy()
        else:
            done = batch["done"]

        if next_state_value is None:
            assert np.isclose(gae_lambda, 1.0)
            next_state_value = np.zeros_like(rew)
        else:
            # mask next_state_value
            next_state_value = next_state_value * (1.0 - done)

        state_value = (
            np.roll(next_state_value, 1) if state_value is None else state_value
        )

        # XXX(ming): why we clip the unfinished index?
        # end_flag = batch.done.copy()
        # truncated
        # end_flag[np.isin(indices, buffer.unfinished_index())] = True
        if gae_lambda == 0:
            returns = rew + gamma * next_state_value
        else:
            advantage = Postprocessor.gae_return(
                state_value,
                next_state_value,
                rew,
                done,
                gamma,
                gae_lambda,
            )
            returns = advantage + state_value
        # normalization varies from each policy, so we don't do it here
        return returns, advantage
