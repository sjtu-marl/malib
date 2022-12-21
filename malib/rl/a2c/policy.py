# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

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
            values = self.critic(observation)
        return values.cpu().numpy()
