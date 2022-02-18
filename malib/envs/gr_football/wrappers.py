import gym
import numpy as np

from malib.utils.typing import Callable, AgentID, Any, Dict

from malib.envs.env import GroupWrapper


def GroupedGFBall(base_env, parameter_sharing_func: Callable):
    class PSGFootBall(GroupWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.is_sequential = False

        def build_state_from_observation(
            self, agent_observation: Dict[AgentID, Any]
        ) -> Dict[AgentID, Any]:
            # considering overlapping in the mapping, maybe?
            group_obs = {gid: [] for gid in self.groups}
            for gid, container in group_obs.items():
                container.extend(
                    [agent_observation[aid] for aid in self.group_to_agents[gid]]
                )
                container = np.vstack(container)
            # share among agents
            return {
                aid: group_obs[self.group_rule(aid)] for aid in self.possible_agents
            }

        def build_state_spaces(self) -> Dict[AgentID, gym.Space]:
            return {
                aid: gym.spaces.Tuple(
                    [
                        self.observation_spaces[member]
                        for member in self.group_to_agents[self.group_rule(aid)]
                    ]
                )
                for aid in self.possible_agents
            }

        def group_rule(self, agent_id: AgentID) -> str:
            return parameter_sharing_func(agent_id)

    return PSGFootBall(base_env)
