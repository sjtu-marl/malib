import copy

from typing import Dict, Any


POLICY_CONFIG = {
    "name": None,
    "action_space": None,
    "observation_space": None,
    "model_config": None,
    "custom_config": None,
}


ROLLOUT_CONFIG = {
    # use sync sampler or async sampler
    "sample_sync": True,
    # -1: unfixed length, or other integer larger than 0
    "fragment_length": 100,
    # environments for each worker
    "num_envs": 1,
    "batch_mode": "truncated_episodes",
}

PARAMETER_SERVER_CONFIG = {
    # ...
    "maxsize": 1
}

OFFLINE_DATASET_SERVER_CONFIG = {
    # ...
    "maxsize": 1
}

DATA_EXCHANGER_CONFIG = {
    "parameter_server": PARAMETER_SERVER_CONFIG,
    "dataset_server": OFFLINE_DATASET_SERVER_CONFIG,
    "coordinator": None,
}

ENV_CONFIG = {}

ROLLOUT_MANAGER_CONFIG = {
    "rollout": ROLLOUT_CONFIG,
    "policy": POLICY_CONFIG,
    # size of task queue
    "max_task_size": 1,
    # number of workers
    "worker_num": 1,
    "env": ENV_CONFIG,
}

EXPERIMENT_MANAGER_CONFIG = {
    "primary": "experiment",
    "secondary": "run",
    "key": 0,
    "nid": 0,
    "log_level": 70,
}


def update_config(ori_config, kwargs: Dict[str, Any]):
    """ Rewrite original configuration with given keys """
    # check keys
    legal_keys = set(ori_config.keys())
    update_keys = set(kwargs.keys())
    if legal_keys >= update_keys:
        tmp = copy.copy(ori_config)
        tmp.update(kwargs)
        return tmp
    else:
        raise KeyError(f"Illegal keys found in update_keys: {list(kwargs.keys())}")
