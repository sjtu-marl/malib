from typing import Any

import gym
import torch
import numpy as np

from torch.distributions import Categorical, Normal

from malib.algorithm.common.policy import Policy
from malib.utils.episode import EpisodeKey
from malib.utils.typing import DataTransferType, Dict, Tuple, BehaviorMode
from malib.algorithm.common.model import get_model
from malib.algorithm.common import misc


class DiscreteSAC(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(DiscreteSAC, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        assert isinstance(action_space, gym.spaces.Discrete), action_space

        if self.custom_config.get("use_auto_alpha", False):
            self.use_auto_alpha = True
            self._target_entropy = 0.98 * np.log(np.prod(action_space.n))
            self._log_alpha = torch.zeros(1, requires_grad=True)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self.use_auto_alpha = False
            self._alpha = self.custom_config.get("alpha", 0.05)

        self.set_actor(
            get_model(self.model_config.get("actor"))(
                observation_space, action_space, self.custom_config["use_cuda"]
            )
        )

        # set two critics
        self.set_critic(
            *[
                get_model(self.model_config.get("critic"))(
                    observation_space,
                    action_space,
                    self.custom_config["use_cuda"],
                )
                for _ in range(2)
            ]
        )
        self.target_actor = get_model(self.model_config.get("actor"))(
            observation_space, action_space, self.custom_config["use_cuda"]
        )
        self.target_critic_1 = get_model(self.model_config.get("critic"))(
            observation_space, action_space, self.custom_config["use_cuda"]
        )
        self.target_critic_2 = get_model(self.model_config.get("critic"))(
            observation_space, action_space, self.custom_config["use_cuda"]
        )

        self.register_state(self.actor, "actor")
        self.register_state(self.critic_1, "critic_1")
        self.register_state(self.critic_2, "critic_2")
        self.register_state(self.target_actor, "target_actor")
        self.register_state(self.target_critic_1, "target_critic_1")
        self.register_state(self.target_critic_2, "target_critic_2")

        self.update_target()

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        logits = self.actor(observation)
        m = Categorical(logits=logits)
        actions = m.sample()
        return actions

    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[Any, Any, Any]:
        logits = self.actor(observation)

        assert len(logits.shape) > 1, logits.shape
        if "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"]).to(logits.device)
        else:
            mask = torch.ones_like(logits, device=logits.device, dtype=logits.dtype)
        logits = logits * mask
        assert len(logits.shape) > 1, logits.shape
        m = Categorical(logits=logits)
        action_probs = m.probs
        # actions = torch.argmax(action_probs, dim=-1, keepdim=True).detach()
        actions = m.sample().detach()

        return (
            actions.cpu().numpy(),
            action_probs.detach().cpu().numpy(),
            kwargs[EpisodeKey.RNN_STATE]
        )

    def compute_actions_by_target_actor(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        with torch.no_grad():
            pi = self.target_actor(observation)
        return pi

    def update_target(self):
        self.target_critic_1.load_state_dict(self._critic_1.state_dict())
        self.target_critic_2.load_state_dict(self._critic_2.state_dict())
        self.target_actor.load_state_dict(self._actor.state_dict())

    def set_critic(self, critic_1, critic_2):
        self._critic_1 = critic_1
        self._critic_2 = critic_2

    @property
    def critic_1(self):
        return self._critic_1

    @property
    def critic_2(self):
        return self._critic_2

    def soft_update(self, tau=0.01):
        misc.soft_update(self.target_critic_1, self.critic_1, tau)
        misc.soft_update(self.target_critic_2, self.critic_2, tau)
        misc.soft_update(self.target_actor, self.actor, tau)
