from collections import defaultdict
import importlib
import supersuit
import gym

from malib.envs.vector_env import VectorEnv
from malib.utils.typing import (
    Dict,
    Any,
    List,
    Sequence,
    AgentID,
)
from malib.backend.datapool.offline_dataset_server import Episode


def nested_env_creator(ori_creator: type, wrappers: Sequence[Dict]) -> type:
    """Wrap original atari environment creator with multiple wrappers"""

    def creator(**env_config):
        env = ori_creator(**env_config)
        # parse wrappers
        for wconfig in wrappers:
            name = wconfig["name"]
            params = wconfig["params"]

            wrapper = getattr(
                supersuit, name
            )  # importlib.import_module(f"supersuit.{env_desc['wrapper']['name']}")

            if isinstance(params, Sequence):
                env = wrapper(env, *params)
            elif isinstance(params, Dict):
                env = wrapper(env, **params)
            else:
                raise TypeError(f"Unexpected type: {type(params)}")
        return env

    return creator


def make_env(env_id, parallel=True, **env_configs) -> Any:
    env_module = env_module = importlib.import_module(f"pettingzoo.atari.{env_id}")
    ori_caller = env_module.env if not parallel else env_module.parallel_env
    wrappers = (
        env_configs.pop("wrappers") if env_configs.get("wrappers") is not None else []
    )
    wrapped_caller = nested_env_creator(ori_caller, wrappers)

    return wrapped_caller(**env_configs)
