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

from typing import Dict, List, Any, Union, Tuple, Sequence

import uuid
import gym
import numpy as np

from malib.utils.typing import AgentID
from malib.utils.general import flatten_dict


class Environment:
    def __init__(self, **configs):
        self.is_sequential = False
        self.episode_metrics = {
            "env_step": 0,
            "episode_reward": 0.0,
            "agent_reward": {},
            "agent_step": {},
        }

        self.runtime_id = uuid.uuid4().hex
        # -1 means no horizon limitation
        self.max_step = 1000
        self.cnt = 0
        self.episode_meta_info = {"max_step": self.max_step}

        if configs.get("custom_metrics") is not None:
            self.custom_metrics = configs.pop("custom_metrics")
        else:
            self.custom_metrics = {}

        self._configs = configs
        self._current_players = []
        self._state: Dict[str, np.ndarray] = None

    def record_episode_info_step(
        self,
        state: Any,
        observations: Dict[AgentID, Any],
        rewards: Dict[AgentID, Any],
        dones: Dict[AgentID, bool],
        infos: Any,
    ):
        """Analyze timestep and record it as episode information.

        Args:
            state (Any): Environment state.
            observations (Dict[AgentID, Any]): A dict of agent observations
            rewards (Dict[AgentID, Any]): A dict of agent rewards.
            dones (Dict[AgentID, bool]): A dict of done signals.
            infos (Any): Information.
        """

        reward_ph = self.episode_metrics["agent_reward"]
        step_ph = self.episode_metrics["agent_step"]
        for aid, r in rewards.items():
            if aid not in reward_ph:
                reward_ph[aid] = []
                step_ph[aid] = 0
            reward_ph[aid].append(r)
            step_ph[aid] += 1
        self.episode_meta_info["env_done"] = dones["__all__"]
        self.episode_metrics["env_step"] += 1
        self.episode_metrics["episode_reward"] += sum(rewards.values())

    @property
    def possible_agents(self) -> List[AgentID]:
        """Return a list of environment agent ids"""

        raise NotImplementedError

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        """A dict of agent observation spaces"""

        raise NotImplementedError

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        """A dict of agent action spaces"""

        raise NotImplementedError

    def reset(self, max_step: int = None) -> Union[None, Sequence[Dict[AgentID, Any]]]:
        """Reset environment and the episode info handler here."""

        self.max_step = max_step or self.max_step
        self.cnt = 0

        self.episode_metrics = {
            "env_step": 0,
            "episode_reward": 0.0,
            "agent_reward": {k: [] for k in self.possible_agents},
            "agent_step": {k: 0.0 for k in self.possible_agents},
        }
        self.episode_meta_info.update(
            {
                "max_step": self.max_step,
                "env_done": False,
            }
        )

    def env_done_check(self, agent_dones: Dict[AgentID, bool]) -> bool:
        # default by any
        done1 = any(agent_dones.values())
        # self.max_step == -1 means no limits
        done2 = self.cnt >= self.max_step > 0
        return done1 or done2

    def step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Any,
    ]:
        """Return a 5-tuple as (state, observation, reward, done, info). Each item is a dict maps from agent id to entity.

        Note:
            If state return of this environment is not activated, the return state would be None.

        Args:
            actions (Dict[AgentID, Any]): A dict of agent actions.

        Returns:
            Tuple[ Dict[AgentID, Any], Dict[AgentID, Any], Dict[AgentID, float], Dict[AgentID, bool], Any]: A tuple follows the order as (state, observation, reward, done, info).
        """

        self.cnt += 1
        rets = list(self.time_step(actions))
        rets[3]["__all__"] = self.env_done_check(rets[3])
        if rets[3]["__all__"]:
            rets[3] = {k: True for k in rets[3].keys()}
        rets = tuple(rets)
        self.record_episode_info_step(*rets)
        # state, obs, reward, done, info.
        return rets

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        """Environment stepping logic.

        Args:
            actions (Dict[AgentID, Any]): Agent action dict.

        Raises:
            NotImplementedError: Not implmeneted error

        Returns:
            Tuple[Dict[AgentID, Any], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Any]]: A 4-tuples, listed as (observations, rewards, dones, infos)
        """

        raise NotImplementedError

    @staticmethod
    def action_adapter(policy_outputs: Dict[str, Dict[AgentID, Any]], **kwargs):
        """Convert policy action to environment actions. Default by policy action"""

        return policy_outputs["action"]

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed: int = None):
        pass

    def collect_info(self) -> Dict[str, Any]:
        # flatten metrics
        res1 = flatten_dict(self.episode_metrics)
        res2 = flatten_dict(self.custom_metrics)
        return {**res1, **res2}


class Wrapper(Environment):
    """Wraps the environment to allow a modular transformation"""

    def __init__(self, env: Environment):
        """Construct a wrapper for a given enviornment instance.

        Args:
            env (Environment): Environment instance.
        """

        self.env = env
        self.max_step = self.env.max_step
        self.cnt = self.env.cnt
        self.episode_meta_info = self.env.episode_meta_info
        self.custom_metrics = self.env.custom_metrics
        self.runtime_id = self.env.runtime_id
        self.is_sequential = self.env.is_sequential
        self.episode_metrics = self.env.episode_metrics

    @property
    def possible_agents(self) -> List[AgentID]:
        return self.env.possible_agents

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self.env.action_spaces

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self.env.observation_spaces

    def step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Any,
    ]:
        return self.env.step(actions)

    def close(self):
        return self.env.close()

    def seed(self, seed: int = None):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, max_step: int = None) -> Union[None, Tuple[Dict[AgentID, Any]]]:
        ret = self.env.reset(max_step)
        return ret

    def collect_info(self) -> Dict[str, Any]:
        return self.env.collect_info()


class GroupWrapper(Wrapper):
    def __init__(
        self,
        env: Environment,
        aid_to_gid: Dict[AgentID, str],
        agent_groups: Dict[str, List[AgentID]],
    ):
        super(GroupWrapper, self).__init__(env)
        self._aid_to_gid = aid_to_gid
        self._agent_groups = agent_groups
        self._state_spaces = self.build_state_spaces()

    @property
    def state_spaces(self) -> Dict[str, gym.Space]:
        """Return a dict of group state spaces.

        Note:
            Users must implement the method `build_state_space`.

        Returns:
            Dict[str, gym.Space]: A dict of state spaces.
        """

        return self._state_spaces

    @property
    def possible_agents(self) -> List[str]:
        return list(self.agent_groups.keys())

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        raise NotImplementedError

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return NotImplementedError

    @property
    def agent_groups(self) -> Dict[str, List[AgentID]]:
        return self._agent_groups

    def agent_to_group(self, agent_id: AgentID) -> str:
        """Mapping agent id to groupd id.

        Args:
            agent_id (AgentID): Agent id.

        Returns:
            str: Group id.
        """

        return self._aid_to_gid[agent_id]

    def build_state_spaces(self) -> Dict[str, gym.Space]:
        """Call `self.group_to_agents` to build state space here"""

        raise NotImplementedError

    def build_state_from_observation(
        self, agent_observation: Dict[AgentID, Any]
    ) -> Dict[str, np.ndarray]:
        """Build state from raw observation.

        Args:
            agent_observation (Dict[AgentID, Any]): A dict of agent observation.

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            Dict[str, np.ndarray]: A dict of states.
        """

        raise NotImplementedError

    def reset(self, max_step: int = None) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        rets = super(GroupWrapper, self).reset(max_step=max_step)
        state = self.build_state_from_observation(rets[0])
        self.set_state(state)
        observations = rets[0]
        grouped_obs = {
            gid: tuple(observations[aid] for aid in agents)
            for gid, agents in self.agent_groups.items()
        }
        grouped_action_masks = self.action_mask_extract(grouped_obs)
        # FIXME(ming): return states and obs, not obs and masks
        return (grouped_obs, grouped_action_masks)

    def action_mask_extract(self, raw_observations: Dict[str, Any]):
        action_masks = {}
        for gid, agent_obs_tup in raw_observations.items():
            if isinstance(agent_obs_tup[0], dict) and "action_mask" in agent_obs_tup[0]:
                action_masks[gid] = tuple(x["action_mask"] for x in agent_obs_tup)
        return action_masks

    def record_episode_info_step(self, observations, rewards, dones, infos):
        reward_ph = self.episode_metrics["agent_reward"]
        step_ph = self.episode_metrics["agent_step"]
        for aid, r in rewards.items():
            if aid not in reward_ph:
                reward_ph[aid] = []
                step_ph[aid] = 0
            reward_ph[aid].append(r)
            step_ph[aid] += 1
        self.episode_meta_info["env_done"] = dones["__all__"]
        self.episode_metrics["env_step"] += 1
        self.episode_metrics["episode_reward"] += sum(map(sum, rewards.values()))

    def env_done_check(self, agent_dones: Dict[AgentID, bool]) -> bool:
        # default by any
        done1 = any(map(any, agent_dones.values()))
        done2 = self.cnt >= self.max_step > 0
        return done1 or done2

    def time_step(self, actions: Dict[str, Any]):
        agent_actions = {}
        for gid, _actions in actions.items():
            agent_actions.update(dict(zip(self.agent_groups[gid], _actions)))
        rets = self.env.time_step(agent_actions)
        state = self.build_state_from_observation(rets[0])
        self.set_state(state)
        # regroup obs
        observations = rets[0]
        rewards = rets[1]
        dones = rets[2]
        infos = rets[3]

        grouped_obs = {
            gid: tuple(observations[aid] for aid in agents)
            for gid, agents in self.agent_groups.items()
        }
        grouped_rewards = {
            gid: tuple(rewards[agent] for agent in agents)
            for gid, agents in self.agent_groups.items()
        }
        grouped_dones = {
            gid: tuple(dones[agent] for agent in agents)
            for gid, agents in self.agent_groups.items()
        }
        grouped_infos = {
            gid: tuple(infos[agent] for agent in agents)
            for gid, agents in self.agent_groups.items()
        }
        return grouped_obs, grouped_rewards, grouped_dones, grouped_infos
