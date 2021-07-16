from functools import reduce
from operator import mul
from typing import Dict, Any

import gym
import numpy as np
import torch

from malib.algorithm.common import misc
from malib.algorithm.common.policy import Policy
from malib.algorithm.common.model import get_model
from malib.utils.typing import DataTransferType, BehaviorMode, EvaluateResult
from malib.backend.datapool.offline_dataset_server import Episode


class DQN(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(DQN, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        assert isinstance(action_space, gym.spaces.Discrete)

        self._gamma = custom_config.get("gamma", 0.98)
        self._eps_min = custom_config.get("eps_min", 1e-2)
        self._eps_max = custom_config.get("eps_max", 1.0)
        self._eps_decay = custom_config.get("eps_decay", 2000)

        if self._eps_decay <= 1.0:
            # convert to decay step
            self._eps_decay = int((self._eps_max - self._eps_min) / self._eps_decay)

        self._model = get_model(self.model_config["critic"])(
            observation_space, action_space, self.custom_config.get("use_cuda", False)
        )
        self._target_model = get_model(self.model_config["critic"])(
            observation_space, action_space, self.custom_config.get("use_cuda", False)
        )

        self._step = 0

        self.register_state(self._gamma, "_gamma")
        self.register_state(self._eps_max, "_eps_max")
        self.register_state(self._eps_min, "_eps_min")
        self.register_state(self._eps_decay, "_eps_decay")
        self.register_state(self._model, "critic")
        self.register_state(self._target_model, "target_critic")
        self.register_state(self._step, "_step")
        self.set_critic(self._model)
        self.target_critic = self._target_model

        self.target_update()

    def target_update(self):
        with torch.no_grad():
            misc.hard_update(self.target_critic, self.critic)

    def _calc_eps(self):
        return self._eps_min + (self._eps_max - self._eps_min) * np.exp(
            -self._step / self._eps_decay
        )

    def compute_action(self, observation: DataTransferType, **kwargs):
        """Compute action with one piece of observation. Behavior mode is used to do exploration/exploitation trade-off.

        :param DataTransferType observation: Transformed observation in numpy.ndarray format.
        :param dict kwargs: Optional dict-style arguments. {behavior_mode: ..., others: ...}
        :return: A tuple of action, action distribution and extra_info.
        """

        behavior = kwargs.get("behavior_mode", BehaviorMode.EXPLORATION)
        batch_size = len(observation)
        logits = torch.softmax(self.critic(observation), dim=-1)

        # do masking
        if "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"]).to(logits.device)
        else:
            mask = torch.ones(logits.shape, device=logits.device, dtype=logits.dtype)
        assert mask.shape == logits.shape, (mask.shape, logits.shape)

        action_probs = misc.masked_softmax(logits, mask)
        m = torch.distributions.Categorical(probs=action_probs)

        if behavior == BehaviorMode.EXPLORATION:
            if np.random.random() < self._calc_eps():
                actions = m.sample().view((-1, 1))
                return (
                    actions.to("cpu").numpy(),
                    action_probs.detach().to("cpu").numpy(),
                    {Episode.ACTION_DIST: action_probs.detach().to("cpu").numpy()},
                )

        actions = torch.argmax(action_probs, dim=-1, keepdim=True)
        extra_info = {Episode.ACTION_DIST: action_probs.detach().to("cpu").numpy()}

        return (
            actions.detach().numpy(),
            action_probs.detach().to("cpu").numpy(),
            extra_info,
        )

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        raise NotImplementedError

    def soft_update(self, tau=0.01):
        with torch.no_grad():
            misc.soft_update(self.target_critic, self.critic, tau)
