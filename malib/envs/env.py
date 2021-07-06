import gym

from malib.utils.typing import Dict, AgentID, List, Any, Union, Tuple


class Environment:
    def __init__(self, **configs):
        self.is_sequential = False
        self._extra_returns = []
        self._trainable_agents = None
        self._configs = configs
        self._env = None

    @staticmethod
    def from_sequential_game(env, **kwargs):
        _env = Environment(**kwargs)
        _env._env = env
        _env._trainable_agents = env.possible_agents
        _env.is_sequential = True
        return _env

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def trainable_agents(self) -> Union[Tuple, None]:
        """Return trainble agents, if registered return a tuple, otherwise None"""
        return self._trainable_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._env.observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._env.action_spaces

    @property
    def extra_returns(self):
        return self._extra_returns

    def reset(self, *args, **kwargs):
        return self._env.reset()

    def step(self, actions: Dict[AgentID, Any]):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        self._env.close()
