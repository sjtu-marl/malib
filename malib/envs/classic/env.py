import importlib

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode


class ClassicEnv(Environment):
    def __init__(self, **configs):
        super(ClassicEnv, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})

        env_module = importlib.import_module(f"pettingzoo.classic.{env_id}")
        ori_caller = env_module.env

        self.is_sequential = True
        self._env = ori_caller(**scenario_configs)
        self._trainable_agents = self._env.possible_agents

        self._extra_returns = [Episode.ACTION_MASK]

    def step(self, actions: Dict[AgentID, Any]):
        raise NotImplementedError("Sequential game does not support parallel stepping!")
        # return super().step(actions)

    def render(self, *args, **kwargs):
        pass

    def reset(self):
        return self._env.reset()
