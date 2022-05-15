from typing import Dict, Any, List, Union

import gym

from gym import spaces

from malib.utils.typing import AgentID
from malib.rollout.envs import Environment


class DummyEnv(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        self.num_agents = configs.get("num_agents", 2)
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3, 2))

    @property
    def possible_agents(self) -> List[AgentID]:
        return [f"dummy_{i}" for i in range(self.num_agents)]

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return {agent: self.action_space for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return {agent: self.observation_space for agent in self.possible_agents}

    def time_step(self, actions: Dict[AgentID, Any]):
        observations = {
            agent: obs_space.sample()
            for agent, obs_space in self.observation_spaces.items()
        }
        rewards = {agent: 0.0 for agent in self.possible_agents}
        done = self.cnt >= self.max_step
        dones = dict.fromkeys(self.possible_agents, done)
        infos = dict.fromkeys(self.possible_agents, {})
        return observations, rewards, dones, infos

    def close(self):
        pass

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super().reset(max_step, custom_reset_config)
        observations = {
            agent: obs_space.sample()
            for agent, obs_space in self.observation_spaces.items()
        }
        return (observations,)
