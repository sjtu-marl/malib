from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from gym import spaces

from malib.rl.pg import PGPolicy
from malib.models.torch import continuous, discrete


class A2CPolicy(PGPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )

        preprocess_net: nn.Module = self.actor.preprocess
        if isinstance(action_space, spaces.Discrete):
            self.critic = discrete.Critic(
                preprocess_net=preprocess_net,
                hidden_sizes=model_config["hidden_sizes"],
                device=self.device,
            )
        elif isinstance(action_space, spaces.Box):
            self.critic = continuous.Critic(
                preprocess_net=preprocess_net,
                hidden_sizes=model_config["hidden_sizes"],
                device=self.device,
            )
        else:
            raise TypeError(
                "Unexpected action space type: {}".format(type(action_space))
            )

        self.register_state(self.critic, "critic")

    def value_function(self, observation: torch.Tensor, evaluate: bool, **kwargs):
        """Compute values of critic."""

        with torch.no_grad():
            values, _ = self.critic(observation)
        return values.cpu().numpy()
