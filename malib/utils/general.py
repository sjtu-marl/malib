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

from typing import (
    OrderedDict,
    Dict,
    List,
    Union,
    Optional,
    TypeVar,
    Callable,
    Tuple,
    Any,
)
from collections import deque
from collections.abc import Mapping, Sequence

import copy

import torch
import numpy as np

from malib import settings


T = TypeVar("T")


def update_rollout_configs(
    global_dict: Dict[str, Any], runtime_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update default rollout configuration and return a new one.

    Note:
        the keys in rollout configuration include 
        - `num_threads`: int, the total threads in a rollout worker to run simulations.
        - `num_env_per_thread`: int, indicate how many environment will be created for \
            each running thread.
        - `batch_mode`: default by 'time_step'.
        - `post_processor_types`: default by ['default'].
        - `use_subprov_env`: use sub proc environment or not, default by False.
        - `num_eval_threads`: the number of threads for evaluation, default by 1.

    Args:
        global_dict (Dict[str, Any]): The default global configuration.
        runtime_dict (Dict[str, Any]): The default global configuration.

    Returns:
        Dict[str, Any]: Updated rollout configuration.
    """

    new_instance = global_dict["rollout_worker"].copy()
    rollout_config = runtime_dict.get("rollout_worker", {})
    for k, v in rollout_config.items():
        new_instance[k] = v

    defaults = {
        "batch_mode": "time_step",
        "postprocessor_types": ["default"],
        "use_subproc_env": False,
        "num_eval_threads": 1,
    }

    for k, v in defaults.items():
        if new_instance.get(k) is None:
            new_instance[k] = v
    return new_instance


def update_training_config(
    global_dict: Dict[str, Any], runtime_dict: Dict[str, Any]
) -> Dict[str, Any]:
    training_config = global_dict["training"].copy()
    env_description = runtime_dict["env_description"]
    runtime_training = runtime_dict["training"]

    training_config["interface"].update(runtime_training["interface"])

    # check None
    if training_config["interface"].get("observation_spaces") is None:
        training_config["interface"]["observation_spaces"] = env_description[
            "observation_spaces"
        ]
    if training_config["interface"].get("action_spaces") is None:
        training_config["interface"]["action_spaces"] = env_description["action_spaces"]
    if training_config["interface"]["use_init_policy_pool"]:
        assert runtime_dict["task_mode"] == "gt"

    training_config["config"].update(runtime_dict["training"]["config"])
    return training_config


def update_dataset_config(global_dict: Dict[str, Any], runtime_config: Dict[str, Any]):
    dataset_config = global_dict["dataset"].copy()
    dataset_config.update(runtime_config.get("dataset", {}))
    return dataset_config


def update_parameter_server_config(
    global_dict: Dict[str, Any], runtime_config: Dict[str, Any]
):
    pconfig = global_dict["parameter_server"].copy()
    pconfig.update(runtime_config.get("parameter_server", {}))
    return pconfig


def update_global_evaluator_config(
    global_dict: Dict[str, Any], runtime_config: Dict[str, Any]
):
    econfig = global_dict["global_evaluator"].copy()
    econfig.update(runtime_config.get("global_evaluator", {}))
    return econfig


def update_evaluation_config(
    global_dict: Dict[str, Any], runtime_config: Dict[str, Any]
):
    econfig = global_dict["evaluation"].copy()
    econfig.update(runtime_config.get("evaluation", {}))
    return econfig


def update_configs(runtime_config: Dict[str, Any]):
    """Update global configs with a given dict"""

    assert runtime_config.get("task_mode") in [
        "gt",
        "marl",
    ], "Illegal task mode: {}".format(runtime_config.get("task_mode"))

    rollout_config = update_rollout_configs(settings.DEFAULT_CONFIG, runtime_config)
    env_description = runtime_config["env_description"]
    training_config = update_training_config(settings.DEFAULT_CONFIG, runtime_config)
    algorithms = runtime_config["algorithms"]
    agent_mapping_func = runtime_config.get(
        "agent_mapping_func", settings.DEFAULT_CONFIG["agent_mapping_func"]
    )
    dataset_config = update_dataset_config(settings.DEFAULT_CONFIG, runtime_config)
    parameter_server_config = update_parameter_server_config(
        settings.DEFAULT_CONFIG, runtime_config
    )
    global_evaluator_config = update_global_evaluator_config(
        settings.DEFAULT_CONFIG, runtime_config
    )
    evaluation_config = update_evaluation_config(
        settings.DEFAULT_CONFIG, runtime_config
    )

    return {
        "env_description": env_description,
        "rollout": rollout_config,
        "training": training_config,
        "algorithms": algorithms,
        "agent_mapping_func": agent_mapping_func,
        "task_mode": runtime_config["task_mode"],
        "evaluation": evaluation_config,
        "dataset": dataset_config,
        "global_evaluator": global_evaluator_config,
        "parameter_server": parameter_server_config,
    }


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
    elif dtype in bool:
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


def frozen_data(data):
    _hash = 0
    if isinstance(data, Dict):
        for k, v in data.items():
            _v = frozen_data(v)
            _hash ^= hash((k, _v))
    elif isinstance(data, (List, Tuple)):
        for e in data:
            _hash ^= hash(e)
    else:
        return hash(data)
    return _hash


# =============== below refer to: https://docs.ray.io/en/releases-1.9.1/_modules/ray/util/ml_utils/dict.html#merge_dicts
def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Args:
        d1 (dict): Dict 1, the original dict template.
        d2 (dict): Dict 2, the new dict used to udpate.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: str = False,
    allow_new_subkey_list: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None,
) -> dict:
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if (
                k in override_all_if_type_changes
                and "type" in value
                and "type" in original[k]
                and value["type"] != original[k]["type"]
            ):
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def flatten_dict(
    dt: Dict,
    delimiter: str = "/",
    prevent_delimiter: bool = False,
    flatten_list: bool = False,
):
    """Flatten dict.

    Output and input are of the same dict type.
    Input dict remains the same after the operation.
    """

    def _raise_delimiter_exception():
        raise ValueError(
            f"Found delimiter `{delimiter}` in key when trying to flatten "
            f"array. Please avoid using the delimiter in your specification."
        )

    dt = copy.copy(dt)
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        _raise_delimiter_exception()

    while_check = (dict, list) if flatten_list else dict

    while any(isinstance(v, while_check) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
            elif flatten_list and isinstance(value, list):
                for i, v in enumerate(value):
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(i)])] = v
                remove.append(key)

        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def unflatten_dict(dt: Dict[str, T], delimiter: str = "/") -> Dict[str, T]:
    """Unflatten dict. Does not support unflattening lists."""
    dict_type = type(dt)
    out = dict_type()
    for key, val in dt.items():
        path = key.split(delimiter)
        item = out
        for k in path[:-1]:
            item = item.setdefault(k, dict_type())
            if not isinstance(item, dict_type):
                raise TypeError(
                    f"Cannot unflatten dict due the key '{key}' "
                    f"having a parent key '{k}', which value is not "
                    f"of type {dict_type} (got {type(item)}). "
                    "Change the key names to resolve the conflict."
                )
        item[path[-1]] = val
    return out


def unflatten_list_dict(dt: Dict[str, T], delimiter: str = "/") -> Dict[str, T]:
    """Unflatten nested dict and list.

    This function now has some limitations:
    (1) The keys of dt must be str.
    (2) If unflattened dt (the result) contains list, the index order must be
        ascending when accessing dt. Otherwise, this function will throw
        AssertionError.
    (3) The unflattened dt (the result) shouldn't contain dict with number
        keys.

    Be careful to use this function. If you want to improve this function,
    please also improve the unit test. See #14487 for more details.

    Args:
        dt (dict): Flattened dictionary that is originally nested by multiple
            list and dict.
        delimiter (str): Delimiter of keys.

    Example:
        >>> dt = {"aaa/0/bb": 12, "aaa/1/cc": 56, "aaa/1/dd": 92}
        >>> unflatten_list_dict(dt)
        {'aaa': [{'bb': 12}, {'cc': 56, 'dd': 92}]}
    """
    out_type = list if list(dt)[0].split(delimiter, 1)[0].isdigit() else type(dt)
    out = out_type()
    for key, val in dt.items():
        path = key.split(delimiter)

        item = out
        for i, k in enumerate(path[:-1]):
            next_type = list if path[i + 1].isdigit() else dict
            if isinstance(item, dict):
                item = item.setdefault(k, next_type())
            elif isinstance(item, list):
                if int(k) >= len(item):
                    item.append(next_type())
                    assert int(k) == len(item) - 1
                item = item[int(k)]

        if isinstance(item, dict):
            item[path[-1]] = val
        elif isinstance(item, list):
            item.append(val)
            assert int(path[-1]) == len(item) - 1
    return out


def unflattened_lookup(
    flat_key: str, lookup: Union[Mapping, Sequence], delimiter: str = "/", **kwargs
) -> Union[Mapping, Sequence]:
    """
    Unflatten `flat_key` and iteratively look up in `lookup`. E.g.
    `flat_key="a/0/b"` will try to return `lookup["a"][0]["b"]`.
    """
    if flat_key in lookup:
        return lookup[flat_key]
    keys = deque(flat_key.split(delimiter))
    base = lookup
    while keys:
        key = keys.popleft()
        try:
            if isinstance(base, Mapping):
                base = base[key]
            elif isinstance(base, Sequence):
                base = base[int(key)]
            else:
                raise KeyError()
        except KeyError as e:
            if "default" in kwargs:
                return kwargs["default"]
            raise e
    return base
