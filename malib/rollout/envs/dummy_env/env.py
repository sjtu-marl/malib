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

from typing import Dict, Any, List, Union

import time
import gym

from gym import spaces

from malib.utils.typing import AgentID
from malib.rollout.envs.env import Environment


class DummyEnv(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        self.num_agents = configs.get("num_agents", 2)
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3, 2))
        self.step_time_consumption = configs.get("step_time_consumption", 0.1)
        self.enable_env_state = configs.get("enable_env_state", False)

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
        states = (
            {
                agent: obs_space.sample()
                for agent, obs_space in self.observation_spaces.items()
            }
            if self.enable_env_state
            else None
        )
        rewards = {agent: 0.0 for agent in self.possible_agents}
        done = self.cnt >= self.max_step
        dones = dict.fromkeys(self.possible_agents, done)
        infos = dict.fromkeys(self.possible_agents, {})
        # fake time consumption
        # time.sleep(self.step_time_consumption)
        return states, observations, rewards, dones, infos

    def close(self):
        pass

    def reset(self, max_step: int = None) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super().reset(max_step)
        observations = {
            agent: obs_space.sample()
            for agent, obs_space in self.observation_spaces.items()
        }
        state = (
            {
                agent: obs_space.sample()
                for agent, obs_space in self.observation_spaces.items()
            }
            if self.enable_env_state
            else None
        )
        return state, observations
