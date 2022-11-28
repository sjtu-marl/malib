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

from typing import Dict, Any, Union

import logging
import os
import copy

import gym
import torch
import numpy as np

from gym import spaces
from torch import nn

from malib.rl.common import misc
from malib.rl.common.policy import Policy
from malib.models.torch import make_net
from malib.utils.general import merge_dicts

from .config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class DQNPolicy(Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        **kwargs
    ):
        model_config = merge_dicts(DEFAULT_CONFIG["model_config"].copy(), model_config)
        custom_config = merge_dicts(
            DEFAULT_CONFIG["custom_config"].copy(), custom_config
        )
        agent_dimension = 0

        if isinstance(observation_space, spaces.Tuple):
            # it means the input has agent dimension
            agent_dimension = len(observation_space.spaces)
            assert isinstance(action_space, spaces.Tuple)
            observation_space = observation_space.spaces[0]
            action_space = action_space[0]

        super(DQNPolicy, self).__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )

        assert isinstance(action_space, gym.spaces.Discrete)
        self.critic: nn.Module = make_net(
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
            net_type=model_config.get("net_type", None),
            **model_config["config"]
        )

        self.use_cuda = self.custom_config.get("use_cuda", False)
        self.agent_dimension = agent_dimension

        if self.use_cuda:
            self.critic.to("cuda")

        self._eps = 1.0

        self.register_state(self._eps, "_eps")
        self.register_state(self.critic, "critic")

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float):
        self._eps = value

    def compute_action(
        self,
        observation: torch.Tensor,
        act_mask: Union[torch.Tensor, None],
        evaluate: bool,
        hidden_state: Any = None,
        **kwargs
    ):
        """Compute action in rollout stage. Do not support vector mode yet.

        Args:
            observation (DataArray): The observation batched data with shape=(n_batch, obs_shape).
            act_mask (DataArray): The action mask batched with shape=(n_batch, mask_shape).
            evaluate (bool): Turn off exploration or not.
            state (Any, Optional): The hidden state. Default by None.
        """

        with torch.no_grad():
            if self.agent_dimension > 0:
                # reshape to (n_batch * agent_dimension, shape)
                observation = observation.reshape((-1,) + self.preprocessor.shape)
                if act_mask is not None:
                    act_mask = act_mask.reshape(-1, self._action_space.n)
            logits, state = self.critic(observation)

            # do masking, and mute logits noising
            action_probs = misc.gumbel_softmax(logits, mask=act_mask)

        if not evaluate:
            if np.random.random() < self.eps:
                action_probs = (
                    np.ones((len(observation), self._action_space.n))
                    / self._action_space.n
                )
                if act_mask is not None:
                    legal_actions = np.array(
                        [
                            idx
                            for idx in range(self._action_space.n)
                            if act_mask[0][idx] > 0
                        ],
                        dtype=np.int32,
                    )
                    action = np.random.choice(legal_actions, len(observation))
                else:
                    action = np.random.choice(self._action_space.n, len(observation))
                if self.agent_dimension > 0:
                    action = action.reshape(-1, self.agent_dimension)
                    action_probs = action_probs.reshape(
                        -1, self.agent_dimension, self._action_space.n
                    )
                    logits = (
                        logits.reshape(-1, self.agent_dimension, self._action_space.n)
                        .cpu()
                        .numpy()
                    )
                    if state is not None:
                        raise NotImplementedError
                else:
                    logits = logits.cpu().numpy()
                return action, action_probs, logits, state

        action = torch.argmax(action_probs, dim=-1).cpu().numpy()
        if self.agent_dimension > 0:
            action = action.reshape(-1, self.agent_dimension)
            action_probs = (
                action_probs.reshape(-1, self.agent_dimension, self._action_space.n)
                .cpu()
                .numpy()
            )
            logits = (
                logits.reshape(-1, self.agent_dimension, self._action_space.n)
                .cpu()
                .numpy()
            )
            if state is not None:
                raise NotImplementedError
        else:
            action_probs = action_probs.cpu().numpy()
            logits = logits.cpu().numpy()

        return action, action_probs, logits, state

    def parameters(self):
        return {
            "critic": self._critic.parameters(),
        }

    def value_function(
        self, observation: torch.Tensor, evaluate: bool, **kwargs
    ) -> np.ndarray:
        values, _ = self.critic(observation)
        values = values.detach().cpu().numpy()
        if "act_mask" in kwargs:
            act_mask = kwargs["act_mask"]
            values[act_mask] = -1e9
        return values

    def reset(self, **kwargs):
        pass

    def save(self, path, global_step=0, hard: bool = False):
        file_exist = os.path.exists(path)
        if file_exist:
            logger.warning("(dqn) ! detected existing mode with path: {}".format(path))
        if (not file_exist) or hard:
            torch.save(self._critic.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location="cuda" if self.use_cuda else "cpu")
        self._critic.load_state_dict(state_dict)
