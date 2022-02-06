import gym
import numpy as np
from malib.utils.typing import Dict, AgentID
from malib.envs.agent_interface import AgentInterface
from malib.utils.episode import EpisodeKey

from tests.algorithm.utils import build_random_policy


def simple_discrete_agent(aid, observation_space, action_space):
    def act(rets):
        if EpisodeKey.ACTION_MASK in rets:
            action_mask = rets[EpisodeKey.ACTION_MASK][aid]
        else:
            action_mask = None

        if action_mask is None:
            return action_space.sample()
        else:
            indices = np.where(action_mask == 1)[0]
            return np.random.choice(indices)

    return act


def build_dummy_agent_interfaces(
    observation_spaces: Dict[AgentID, gym.Space],
    action_spaces: Dict[AgentID, gym.Space],
) -> Dict[AgentID, AgentInterface]:
    interfaces = {}
    for agent_id, obs_space in observation_spaces.items():
        interfaces[agent_id] = AgentInterface(
            agent_id,
            obs_space,
            action_spaces[agent_id],
            parameter_server=None,
            policies={
                f"policy_{i}": build_random_policy(obs_space, action_spaces[agent_id])
                for i in range(2)
            },
        )
    return interfaces
