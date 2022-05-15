from typing import Dict, Any, Union

import logging
import os
import copy

import gym
import torch
import numpy as np

from torch import nn

from malib.algorithm.common import misc
from malib.algorithm.common.policy import Policy

from malib.models.torch import make_net


logger = logging.getLogger(__name__)


class DQNPolicy(Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        is_fixed: bool = False,
        replacement: Dict = None,
    ):
        super(DQNPolicy, self).__init__(
            observation_space,
            action_space,
            model_config,
            custom_config,
            is_fixed,
            replacement,
        )

        assert isinstance(action_space, gym.spaces.Discrete)

        if replacement is not None:
            self.critic = replacement["critic"]
        else:
            self.critic: nn.Module = make_net(
                observation_space=observation_space,
                action_space=action_space,
                device=self.device,
                net_type=model_config.get("net_type", None),
                **model_config["config"]
            )

        self.use_cuda = self.custom_config.get("use_cuda", False)

        if self.use_cuda:
            self.critic = self.critic.to("cuda")

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
        action_mask: Union[torch.Tensor, None],
        hidden_state: Any,
        evaluate: bool,
        **kwargs
    ):
        """Compute action in rollout stage. Do not support vector mode yet.

        Args:
            observation (DataArray): The observation batched data with shape=(n_batch, *obs_shape).
            action_mask (DataArray): The action mask batched with shape=(n_batch, *mask_shape).
            evaluate (bool): Turn off exploration or not.
            state (Any, Optional): The hidden state. Default by None.
        """

        observation = torch.as_tensor(
            observation, device="cuda" if self.use_cuda else "cpu"
        )

        with torch.no_grad():
            logits, state = self.critic(observation)

            # do masking
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).to(logits.device)
                action_probs = misc.masked_gumbel_softmax(logits, mask)
                assert mask.shape == logits.shape, (mask.shape, logits.shape)
            else:
                action_probs = misc.gumbel_softmax(logits, hard=True)

        if not evaluate:
            if np.random.random() < self.eps:
                action_probs = (
                    np.ones((len(observation), self._action_space.n))
                    / self._action_space.n
                )
                if action_mask is not None:
                    legal_actions = np.array(
                        [
                            idx
                            for idx in range(self._action_space.n)
                            if action_mask[0][idx] > 0
                        ],
                        dtype=np.int32,
                    )
                    action = np.random.choice(legal_actions, len(observation))
                else:
                    action = np.random.choice(self._action_space.n, len(observation))
                return action, action_probs, logits.cpu().numpy(), state

        action = torch.argmax(action_probs, dim=-1).cpu().numpy()
        return action, action_probs.cpu().numpy(), logits.cpy().numpy(), state

    def parameters(self):
        return {
            "critic": self._critic.parameters(),
        }

    def value_function(
        self, observation: torch.Tensor, evaluate: bool, **kwargs
    ) -> np.ndarray:
        states = torch.as_tensor(states, device="cuda" if self.use_cuda else "cpu")
        values = self.critic(states).detach().cpu().numpy()
        if action_mask is not None:
            values[action_mask] = -1e9
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
