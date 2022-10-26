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

from typing import Any, Tuple, Union, Dict

import numpy as np
import torch

from gym import spaces
from torch import nn

from malib.models.torch import net, discrete, continuous
from malib.rl.common import misc
from malib.rl.common.policy import Policy, Action, ActionDist, Logits
from malib.utils.general import merge_dicts
from .config import DEFAULT_CONFIG


class PGPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        **kwargs
    ):
        """Build a REINFORCE policy whose input and output dims are determined by observation_space and action_space, respectively.

        Args:
            observation_space (spaces.Space): The observation space.
            action_space (spaces.Space): The action space.
            model_config (Dict[str, Any]): The model configuration dict.
            custom_config (Dict[str, Any]): The custom configuration dict.
            is_fixed (bool, optional): Indicates fixed policy or trainable policy. Defaults to False.

        Raises:
            NotImplementedError: Does not support other action space type settings except Box and Discrete.
            TypeError: Unexpected action space.
        """

        # update model_config with default ones
        model_config = merge_dicts(DEFAULT_CONFIG["model_config"].copy(), model_config)
        custom_config = merge_dicts(
            DEFAULT_CONFIG["custom_config"].copy(), custom_config
        )

        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )

        # update model preprocess_net config here
        action_shape = (
            (action_space.n,) if len(action_space.shape) == 0 else action_space.shape
        )

        preprocess_net: nn.Module = net.make_net(
            observation_space,
            self.device,
            model_config["preprocess_net"].get("net_type", None),
            **model_config["preprocess_net"]["config"]
        )
        if isinstance(action_space, spaces.Discrete):
            self.actor = discrete.Actor(
                preprocess_net=preprocess_net,
                action_shape=action_shape,
                hidden_sizes=model_config["hidden_sizes"],
                softmax_output=False,
                device=self.device,
            )
        elif isinstance(action_space, spaces.Box):
            self.actor = continuous.Actor(
                preprocess_net=preprocess_net,
                action_shape=action_shape,
                hidden_sizes=model_config["hidden_sizes"],
                max_action=custom_config.get("max_action", 1.0),
                device=self.device,
            )
        else:
            raise TypeError(
                "Unexpected action space type: {}".format(type(action_space))
            )

        self.register_state(self.actor, "actor")

    def value_function(self, observation: torch.Tensor, evaluate: bool, **kwargs):
        """Compute values of critic."""

        return np.zeros((observation.shape[0],), dtype=np.float32)

    def compute_action(
        self,
        observation: torch.Tensor,
        act_mask: Union[torch.Tensor, None],
        evaluate: bool,
        hidden_state: Any = None,
        **kwargs
    ) -> Tuple[Action, ActionDist, Logits, Any]:
        with torch.no_grad():
            logits, hidden = self.actor(observation, state=hidden_state)
            if isinstance(logits, tuple):
                dist = self.dist_fn.proba_distribution(*logits)
            else:
                dist = self.dist_fn.proba_distribution(logits, action_mask=act_mask)
            if evaluate:
                if self.action_type == "discrete":
                    act = misc.masked_logits(logits, mask=act_mask).argmax(-1)
                elif self.action_type == "continuous":
                    act = logits[0]
            else:
                act = dist.sample()
            probs = dist.prob().cpu().numpy()

        logits = logits.cpu().numpy()
        action = act.cpu().numpy()
        action_dist = probs
        state = hidden

        return action, action_dist, logits, state
