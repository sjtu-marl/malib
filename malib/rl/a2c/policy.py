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
from malib.models.torch.net import ActorCritic


class A2CPolicy(PGPolicy):
    def create_model(self):
        # since a PGPolicy creates a model as an Actor.
        actor = super().create_model()

        preprocess_net: nn.Module = actor.preprocess
        if isinstance(self.action_space, spaces.Discrete):
            critic = discrete.Critic(
                preprocess_net=preprocess_net,
                hidden_sizes=self.model_config["hidden_sizes"],
                device=self.device,
            )
        elif isinstance(self.action_space, spaces.Box):
            critic = continuous.Critic(
                preprocess_net=preprocess_net,
                hidden_sizes=self.model_config["hidden_sizes"],
                device=self.device,
            )
        else:
            raise TypeError(
                "Unexpected action space type: {}".format(type(self.action_space))
            )

        return ActorCritic(actor, critic)

    @property
    def actor(self):
        return self.model.actor

    @property
    def critic(self):
        return self.model.critic

    def value_function(self, observation: torch.Tensor, evaluate: bool, **kwargs):
        """Compute values of critic."""

        with torch.no_grad():
            values = self.critic(observation)
        return values.cpu().numpy()
