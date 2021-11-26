from typing import Optional
import gym
import uuid

from dataclasses import dataclass

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

        self.episode_metrics = {"env_step": 0, "agent_reward": {}, "agent_step": {}}
        self.runtime_id = uuid.uuid4().hex
        # -1 means no horizon limitation
        self.max_step = -1
        self.episode_meta_info = {"max_step": self.max_step}

        if configs.get("custom_metrics") is not None:
            self.custom_metrics = configs.pop("custom_metrics")
        else:
            self.custom_metrics = {}

        self._trainable_agents = None
        self._configs = configs
        self._cnt = 0

    def record_episode_info_step(self, rets):
        reward_ph = self.episode_metrics["agent_reward"]
        step_ph = self.episode_metrics["agent_step"]
        for aid, r in rets[EpisodeKey.REWARD].items():
            if aid not in reward_ph:
                reward_ph[aid] = []
                step_ph[aid] = 0
            reward_ph[aid].append(r)
            step_ph[aid] += 1
        self.episode_meta_info["env_done"] = rets[EpisodeKey.DONE]["__all__"]
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

        custom_reset_config = custom_reset_config or {}
        self.episode_metrics = {"env_step": 0, "agent_reward": {}, "agent_step": {}}
        self.max_step = max_step or self.max_step
        self.episode_meta_info.update(
            {
                "max_step": self.max_step,
                "custom_config": custom_reset_config,
                "env_done": False,
            }
        )

    @record_episode_info
    def step(self, actions: Dict[AgentID, Any]):
        """Step inner environment with given agent actions"""

        raise NotImplementedError

    def action_adapter(self, policy_outputs: Dict[str, Dict[AgentID, Any]], **kwargs):
        """Convert policy action to environment actions. Default by policy action"""

        return policy_outputs["action"]

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def collect_info(self) -> Dict[str, Any]:
        return {
            "episode_runtime_info": self.episode_meta_info,
            "episode_metrics": self.episode_metrics,
            "custom_metrics": self.custom_metrics,
        }
