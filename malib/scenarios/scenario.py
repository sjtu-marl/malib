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

from abc import ABC, abstractmethod
from types import LambdaType
from typing import Dict, Any, Set, Tuple
from copy import deepcopy
from collections import defaultdict

import gym

from malib.utils.typing import AgentID
from malib.utils.stopping_conditions import StoppingCondition


DEFAULT_STOPPING_CONDITIONS = {}


def validate_spaces(agent_groups: Dict[str, Set[AgentID]], env_desc: Dict[str, Any]):
    # TODO(ming): check whether the agents in the group share the same observation space and action space
    raise NotImplementedError


def validate_agent_group(
    agent_group: Dict[str, Tuple[AgentID]],
    observation_spaces: Dict[AgentID, gym.Space],
    action_spaces: Dict[AgentID, gym.Space],
) -> None:
    """Validate agent group, check spaces.

    Args:
        agent_group (Dict[str, List[AgentID]]): A dict, mapping from runtime ids to lists of agent ids.
        full_keys (List[AgentID]): A list of original environment agent ids.
        observation_spaces (Dict[AgentID, gym.Space]): Agent observation space dict.
        action_spaces (Dict[AgentID, gym.Space]): Agent action space dict.

    Raises:
        RuntimeError: Agents in a same group should share the same observation space and action space.
        NotImplementedError: _description_
    """
    for agents in agent_group.values():
        select_obs_space = observation_spaces[agents[0]]
        select_act_space = action_spaces[agents[0]]
        for agent in agents[1:]:
            assert type(select_obs_space) == type(observation_spaces[agent])
            assert select_obs_space.shape == observation_spaces[agent].shape
            assert type(select_act_space) == type(action_spaces[agent])
            assert select_act_space.shape == action_spaces[agent].shape


def form_group_info(env_desc, agent_mapping_func):
    agent_groups = defaultdict(lambda: list())
    grouped_obs_space = {}
    grouped_act_space = {}
    for agent in env_desc["possible_agents"]:
        rid = agent_mapping_func(agent)
        agent_groups[rid].append(agent)
        grouped_obs_space[rid] = env_desc["observation_spaces"][agent]
        grouped_act_space[rid] = env_desc["action_spaces"][agent]
    agent_groups = {k: tuple(v) for k, v in agent_groups.items()}
    return {
        "observation_space": grouped_obs_space,
        "action_space": grouped_act_space,
        "agent_groups": agent_groups,
    }


class Scenario(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Any],
        agent_mapping_func: LambdaType,
        training_config: Dict[str, Any],
        rollout_config: Dict[str, Any],
        stopping_conditions: Dict[str, Any],
    ):
        self.name = name
        self.log_dir = log_dir
        self.env_desc = env_desc
        self.algorithms = algorithms
        self.agent_mapping_func = agent_mapping_func
        # then generate grouping information here
        self.group_info = form_group_info(env_desc, agent_mapping_func)
        validate_agent_group(
            self.group_info["agent_groups"],
            env_desc["observation_spaces"],
            env_desc["action_spaces"],
        )
        self.training_config = training_config
        self.rollout_config = rollout_config
        self.stopping_conditions = stopping_conditions or DEFAULT_STOPPING_CONDITIONS

    def copy(self):
        return deepcopy(self)

    @abstractmethod
    def create_global_stopper(self) -> StoppingCondition:
        """Create a global stopper."""

    def with_updates(self, **kwargs) -> "Scenario":
        new_copy = self.copy()
        for k, v in kwargs.items():
            if not hasattr(new_copy, k):
                raise KeyError(f"{k} is not an attribute of {new_copy.__class__}")
            setattr(new_copy, k, v)
        return new_copy
