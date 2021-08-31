from dataclasses import dataclass
import gym

from malib.utils.typing import Dict, AgentID, List, Any, Union, Tuple


@dataclass
class EpisodeInfo:
    total_rewards: Dict[AgentID, float]
    step_cnt: Dict[AgentID, float]

    def __post_init__(self):
        self._extra = {}

    @property
    def extra_info(self) -> Dict[str, Any]:
        return self._extra

    def register_extra(self, **kwargs):
        self._extra.update(kwargs)


class Environment:
    def __init__(self, **configs):
        self.is_sequential = False
        self._extra_returns = []
        self._trainable_agents = None
        self._configs = configs
        self._env = None
        self._total_rewards = {}
        self._cnt = 0
        self.episode_info: EpisodeInfo = None

    def record_episode_info(self, **kwargs):
        rewards = kwargs.get("rewards")

        for agent, reward in rewards.items():
            self.episode_info.total_rewards[agent] += reward
            self.episode_info.step_cnt[agent] = self.cnt

    @property
    def env(self) -> Any:
        return self._env

    @property
    def cnt(self) -> int:
        return self._cnt

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
        self._total_rewards = dict.fromkeys(self._trainable_agents, 0.0)
        self._cnt = 0
        self.episode_info = EpisodeInfo(
            total_rewards=dict.fromkeys(self.possible_agents, 0.0),
            step_cnt=dict.fromkeys(self.possible_agents, 0),
        )
        if kwargs.get("extra_epsiode_info_keys") is not None:
            self.episode_info.register_extra(**kwargs["extra_episode_info_keys"])
        return self._env.reset()

    def step(self, actions: Dict[AgentID, Any], **kwargs):
        self._cnt += 1
        self.record_episode_info(**kwargs)

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        self._env.close()
