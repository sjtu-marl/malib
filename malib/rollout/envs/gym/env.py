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
        return {agent: observations}, {agent: rewards}, {agent: dones}, {agent: infos}

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self, max_step: int = None, custom_reset_config: Dict[str, Any] = None):
        super(GymEnv, self).reset(
            max_step=max_step, custom_reset_config=custom_reset_config
        )

        observation = self._env.reset()
        return {self._default_agent: observation}

    def close(self):
        pass
