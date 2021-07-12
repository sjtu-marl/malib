import logging
import importlib
import supersuit

from malib.envs import Environment
from malib.utils.typing import Dict, Any, Sequence, AgentID
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


class MAAtari(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        env_id = self._configs["env_id"]
        wrappers = self._configs.get("wrappers", [])
        scenario_configs = self._configs.get("scenario_configs", {})
        env_module = env_module = importlib.import_module(f"pettingzoo.atari.{env_id}")
        ori_caller = env_module.parallel_env
        wrapped_caller = nested_env_creator(ori_caller, wrappers)

        self.is_sequential = False
        self._env = wrapped_caller(**scenario_configs)
        self._trainable_agents = self._env.possible_agents

    def step(self, actions: Dict[AgentID, Any]):
        observations, rewards, dones, infos = self._env.step(actions)
        return {
            Episode.NEXT_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            Episode.INFO: infos,
        }

    def render(self, *args, **kwargs):
        pass

    def reset(self):
        observations = self._env.reset()
        return {Episode.CUR_OBS: observations}
