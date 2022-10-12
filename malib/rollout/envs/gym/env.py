# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

from typing import Dict, Any, List, Tuple
import gym

from malib.rollout.envs.env import Environment
from malib.utils.typing import AgentID


class GymEnv(Environment):
    """Single agent gym envrionment"""

    def __init__(self, **configs):
        super(GymEnv, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})

        self.is_sequential = False
        self._env = gym.make(env_id, **scenario_configs)
        self._default_agent = "agent"
        self._observation_spaces = {self._default_agent: self._env.observation_space}
        self._action_spaces = {self._default_agent: self._env.action_space}
        self._trainable_agents = [self._default_agent]

    @property
    def possible_agents(self) -> List[AgentID]:
        return [self._default_agent]

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        observations, rewards, dones, infos = self._env.step(
            actions[self._default_agent]
        )

        # agent done or achieving_max_step_done
        agent = self._default_agent
        return (
            None,
            {agent: observations},
            {agent: rewards},
            {agent: dones},
            {agent: infos},
        )

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self, max_step: int = None):
        super(GymEnv, self).reset(max_step=max_step)

        observation = self._env.reset()
        return None, {self._default_agent: observation}

    def close(self):
        pass
