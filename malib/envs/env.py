from collections import defaultdict

import uuid
import gym
import copy
import numpy as np

from malib.utils.typing import Dict, AgentID, List, Any, Union, Tuple
from malib.utils.episode import EpisodeKey


def record_episode_info(func):
    def wrap(self, actions):
        rets = func(self, actions)
        self.record_episode_info_step(rets)
        return rets

    return wrap


class Environment:
    def __init__(self, **configs):
        self.is_sequential = False

        self.episode_metrics = {"env_step": 0, "reward": {}, "agent_step": {}}
        self.runtime_id = uuid.uuid4().hex
        # -1 means no horizon limitation
        self.max_step = -1
        self.cnt = 0
        self.custom_reset_config = {}
        self.episode_meta_info = {"max_step": self.max_step}

        if configs.get("custom_metrics") is not None:
            self.custom_metrics = configs.pop("custom_metrics")
        else:
            self.custom_metrics = {}

        self._trainable_agents = None
        self._configs = configs

    def record_episode_info_step(self, observations, rewards, dones, infos):
        reward_ph = self.episode_metrics["reward"]
        step_ph = self.episode_metrics["agent_step"]
        for aid, r in rewards.items():
            if aid not in reward_ph:
                reward_ph[aid] = []
                step_ph[aid] = 0
            reward_ph[aid].append(r)
            step_ph[aid] += 1
        self.episode_meta_info["env_done"] = dones["__all__"]
        self.episode_metrics["env_step"] += 1

    @property
    def possible_agents(self) -> List[AgentID]:
        """Return a list of environment agent ids"""

        raise NotImplementedError

    @property
    def trainable_agents(self) -> Union[Tuple, None]:
        """Return trainble agents, if registered return a tuple, otherwise None"""
        return self._trainable_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        """A dict of agent observation spaces"""

        raise NotImplementedError

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        """A dict of agent action spaces"""

        raise NotImplementedError

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        """Reset environment and the episode info handler here."""

        self.max_step = max_step or self.max_step
        self.cnt = 0

        custom_reset_config = custom_reset_config or self.custom_reset_config
        self.episode_metrics = {
            "env_step": 0,
            "reward": {k: [] for k in self.possible_agents},
            "agent_step": {k: 0.0 for k in self.possible_agents},
        }
        self.episode_meta_info.update(
            {
                "max_step": self.max_step,
                "custom_config": copy.deepcopy(custom_reset_config),
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
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
        Dict[AgentID, np.ndarray],
    ]:
        self.cnt += 1
        rets = self.time_step(actions)
        rets[2]["__all__"] = self.env_done_check(rets[EpisodeKey.DONE])
        self.record_episode_info_step(*rets)
        observations = rets[0]
        action_masks = {}
        for agent, obs in observations.items():
            if isinstance(obs, dict) and "action_mask" in obs:
                action_masks[agent] = np.asarray(obs["action_mask"], dtype=np.float32)
        rets = rets + (action_masks,)
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
        return {
            # "episode_runtime_info": self.episode_meta_info,
            "episode_metrics": copy.deepcopy(self.episode_metrics),
            "custom_metrics": copy.deepcopy(self.custom_metrics),
        }


class SequentialEnv(Environment):
    def __init__(self, **configs):
        super(SequentialEnv, self).__init__(**configs)
        self.is_sequential = True
        self.max_step = 2**63

    @property
    def agent_selection(self):
        raise NotImplementedError


class Wrapper(Environment):
    """Wraps the environment to allow a modular transformation"""

    def __init__(self, env: Environment):
        self.env = env
        self.max_step = self.env.max_step
        self.cnt = self.env.cnt
        self.episode_meta_info = self.env.episode_meta_info
        self.custom_reset_config = self.env.custom_reset_config
        self.custom_metrics = self.env.custom_metrics
        self.runtime_id = self.env.runtime_id
        self.is_sequential = self.env.is_sequential
        self.episode_metrics = self.env.episode_metrics

    @property
    def possible_agents(self) -> List[AgentID]:
        return self.env.possible_agents

    @property
    def trainable_agents(self) -> Union[Tuple, None]:
        """Return trainble agents, if registered return a tuple, otherwise None"""
        return self.env.trainable_agents

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self.env.action_spaces

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self.env.observation_spaces

    def time_step(self, actions: Dict[AgentID, Any]):
        return self.env.step(actions)

    def close(self):
        return self.env.close()

    def seed(self, seed: int = None):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        return self.env.reset(max_step, custom_reset_config)

    def collect_info(self) -> Dict[str, Any]:
        return self.env.collect_info()


class GroupWrapper(Wrapper):
    def __init__(self, env: Environment):
        super(GroupWrapper, self).__init__(env)

        self.group_to_agents = defaultdict(lambda: [])
        # self.agent_to_group = {}
        for agent_id in self.possible_agents:
            gid = self.group_rule(agent_id)
            # self.agent_to_group[agent_id] = gid
            self.group_to_agents[gid].append(agent_id)
        # soft frozen group_to_agents
        self.group_to_agents = dict(self.group_to_agents)
        self.groups = tuple(self.group_to_agents.keys())

        self._state_spaces = self.build_state_spaces()

    @property
    def state_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._state_spaces

    def group_rule(self, agent_id: AgentID) -> str:
        """Define the rule of grouping, mapping agent id to group id"""

        raise NotImplementedError

    def build_state_spaces(self) -> Dict[AgentID, gym.Space]:
        """Call `self.group_to_agents` to build state space here"""

        raise NotImplementedError

    def build_state_from_observation(
        self, agent_observation: Dict[AgentID, Any]
    ) -> Dict[AgentID, Any]:
        """Return a dict of state"""

        raise NotImplementedError

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        rets = super(GroupWrapper, self).reset(
            max_step=max_step, custom_reset_config=custom_reset_config
        )
        # add CUR_STATE
        rets[EpisodeKey.CUR_STATE] = self.build_state_from_observation(
            rets[EpisodeKey.CUR_OBS]
        )
        return rets

    def time_step(self, actions: Dict[AgentID, Any]):
        rets = super(GroupWrapper, self).time_step(actions)
        rets[EpisodeKey.NEXT_STATE] = self.build_state_from_observation(
            rets[EpisodeKey.NEXT_OBS]
        )
        return rets
