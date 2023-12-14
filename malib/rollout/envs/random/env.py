from typing import Any, Dict, List, Sequence, Tuple, Union

import gym
import random

from gym import spaces

from malib.rollout.envs.env import Environment
from malib.utils.typing import AgentID


class RandomEnv(Environment):
    def __init__(self, **configs):
        assert "num_agents" in configs
        super().__init__(**configs)

    def register_agents(self):
        return {f"agent_{i}" for i in range(self.configs["num_agents"])}

    def register_observation_spaces(self):
        return {
            agent: spaces.Box(low=-1, high=1, shape=(2,))
            for agent in self.possible_agents
        }

    def register_action_spaces(self):
        return {agent: spaces.Discrete(4) for agent in self.possible_agents}

    def get_state(self) -> Any:
        return None

    def reset(self, max_step: int = None):
        super().reset(max_step)
        obs = {k: v.sample() for k, v in self.observation_spaces.items()}
        return self.get_state(), obs

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        # assert action whether in space
        for k, v in actions.items():
            _space = self.action_spaces[k]
            assert _space.contains(v), (k, v, _space)
        obs = {k: v.sample() for k, v in self.observation_spaces.items()}
        rews = {k: random.random() for k in self.possible_agents}
        state = self.get_state()

        return (
            state,
            obs,
            rews,
            {k: False for k in self.possible_agents},
            {k: {} for k in self.possible_agents},
        )

    def close(self):
        pass
