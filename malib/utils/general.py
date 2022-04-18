from typing import OrderedDict

import torch
import numpy as np
import copy

from malib import settings
from malib.utils.typing import Dict, Callable, List, Tuple, Any
from malib.envs import gen_env_desc


def update_rollout_configs(
    global_dict: Dict[str, Any], runtime_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update default rollout configuration and return a new one.

    :note: the keys in rollout configuration include
        - num_envs, int, the total number of environments for each rollout worker.
        - num_env_per_worker, int.
        - batch_mode, default by 'time_step'.
        - post_processor_types, default by ['default'].
        - use_subprov_env, default by False.
        - num_eval_workers, default by 1.

    :param global_dict: The default global configuration.
    :type global_dict: Dict[str, Any]
    :param runtime_dict: The dict used to update the rollout configuration.
    :type runtime_dict: Dict[str, Any]
    :return: The new rollout configuration.
    :rtype: Dict[str, Any]
    """

    new_instance = global_dict["rollout"].copy()
    rollout_config = runtime_dict.get("config", {})
    for k, v in rollout_config.items():
        new_instance[k] = v

    defaults = {
        "batch_mode": "time_step",
        "postprocessor_types": ["default"],
        "use_subproc_env": False,
        "num_eval_workers": 1,
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

    runtime_config["env_description"] = gen_env_desc(
        runtime_config["env_description"]["creator"]
    )(**runtime_config["env_description"]["config"])
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
