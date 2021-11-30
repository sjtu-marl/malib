from typing import Any

import gym
import numpy as np

import torch
from torch.distributions import Normal

from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType, Dict, Tuple, BehaviorMode
from malib.algorithm.common.model import get_model
from malib.algorithm.common import misc


class SAC(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(SAC, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        assert isinstance(action_space, gym.spaces.Box), action_space

        self.action_squash = self.custom_config.get("action_squash", False)

        self.set_actor(
            get_model(self.model_config.get("actor"))(
                observation_space, action_space, self.custom_config["use_cuda"]
            )
        )

        critic_state_space = gym.spaces.Dict(
            {"obs": observation_space, "act": action_space}
        )
        # set two critics
        self.set_critic(
            *[
                get_model(self.model_config.get("critic"))(
                    critic_state_space,
                    gym.spaces.Discrete(1),
                    self.custom_config["use_cuda"],
                )
                for _ in range(2)
            ]
        )
        self.target_critic_1 = get_model(self.model_config.get("critic"))(
            critic_state_space, gym.spaces.Discrete(1), self.custom_config["use_cuda"]
        )
        self.target_critic_2 = get_model(self.model_config.get("critic"))(
            critic_state_space, gym.spaces.Discrete(1), self.custom_config["use_cuda"]
        )

        self.register_state(self.actor, "actor")
        self.register_state(self.critic_1, "critic_1")
        self.register_state(self.critic_2, "critic_2")
        self.register_state(self.target_critic_1, "target_critic_1")
        self.register_state(self.target_critic_2, "target_critic_2")

        self._eps = np.finfo(np.float32).eps.item()

        self.update_target()

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        logits = self.actor(observation)
        m = Normal(*logits)
        actions = m.sample()
        if self.custom_config["action_squash"]:
            actions = torch.tanh(actions)
        return actions

    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[Any, Any, Any]:
        behavior = kwargs.get("behavior_mode", BehaviorMode.EXPLORATION)

        logits = self.actor(observation)
        print("-------- logits", logits)
        m = Normal(*logits)

        if behavior == BehaviorMode.EXPLORATION:
            actions = m.sample().detach()
        else:
            actions = logits[0]

        log_probs = m.log_prob(actions).sum(dim=-1, keepdim=True)

        if self.custom_config["action_squash"]:
            actions = torch.tanh(actions)
            log_probs = log_probs - torch.log(1 - actions.pow(2) + self._eps).sum(
                -1, keepdim=True
            )

        action_probs = torch.exp(log_probs)

        extra_info = {}
        extra_info["log_probs"] = log_probs.detach().to("cpu").numpy()

        return (
            actions.detach().to("cpu").numpy(),
            action_probs.detach().to("cpu").numpy(),
            kwargs["rnn_state"],
        )

    def update_target(self):
        self.target_critic_1.load_state_dict(self._critic_1.state_dict())
        self.target_critic_2.load_state_dict(self._critic_2.state_dict())

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
