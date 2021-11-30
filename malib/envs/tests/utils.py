import gym

from malib.utils.typing import Dict, AgentID
from malib.envs.agent_interface import AgentInterface
from malib.algorithm.tests import build_random_policy


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
