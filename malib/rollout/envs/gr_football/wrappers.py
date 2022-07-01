from collections import defaultdict
from types import LambdaType
from typing import Any, Dict, List

import gym

from gym import spaces

from malib.utils.typing import AgentID
from malib.utils.preprocessor import get_preprocessor
from malib.rollout.envs.env import GroupWrapper
from .env import GRFootball


class GroupedFootball(GroupWrapper):
    def __init__(self, agent_group_mapping: LambdaType, **config):
        env = GRFootball(**config)
        agent_groups = defaultdict(lambda: [])
        for agent in env.possible_agents:
            gid = agent_group_mapping(agent)
            agent_groups[gid].append(agent)
        agent_groups = dict(agent_groups)
        # build agent to group
        aid_to_gid = {}
        for gid, agents in agent_groups.items():
            aid_to_gid.update(dict.fromkeys(agents, gid))
        self.aid_to_gid = aid_to_gid

        agent_obs_spaces = env.observation_spaces
        agent_act_spaces = env.action_spaces

        self._observation_spaces = {
            gid: spaces.Tuple([agent_obs_spaces[aid] for aid in agents])
            for gid, agents in agent_groups.items()
        }
        self._action_spaces = {
            gid: spaces.Tuple([agent_act_spaces[aid] for aid in agents])
            for gid, agents in agent_groups.items()
        }

        super(GroupedFootball, self).__init__(env, aid_to_gid, agent_groups)

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    def build_state_from_observation(
        self, agent_observation: Dict[AgentID, Any]
    ) -> Dict[AgentID, Any]:
        """Build state spaces for each group.

        Args:
            agent_observation (Dict[AgentID, Any]): A dict of agent observations.

        Returns:
            Dict[AgentID, Any]: A dict of states.
        """

        states = {}

        for gid, agents in self.agent_groups.items():
            selected_obs = tuple(agent_observation[agent] for agent in agents)
            states[gid] = self.state_preprocessors[gid].transform(selected_obs)

        return states

    def build_state_spaces(self) -> Dict[str, gym.Space]:
        state_spaces = {gid: None for gid in self.agent_groups}
        state_preprocessors = {gid: None for gid in self.agent_groups}
        for gid, agents in self.agent_groups.items():
            state_preprocessor = get_preprocessor(self.observation_spaces[gid])(
                self.observation_spaces[gid]
            )
            space_unit = state_preprocessor.observation_space
            state_spaces[gid] = spaces.Tuple([space_unit] * len(agents))
            state_preprocessors[gid] = state_preprocessor
        self.state_preprocessors = state_preprocessors
        return state_spaces
