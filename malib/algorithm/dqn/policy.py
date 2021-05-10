from functools import reduce
from operator import mul
from typing import Dict, Any

import gym
import numpy as np
import torch

from malib.algorithm.common.policy import Policy
from malib.algorithm.common.model import get_model
from malib.utils.typing import DataTransferType, BehaviorMode
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

        self._exploration = True
        self._step = 0

        self.register_state(self._gamma, "gamma")
        self.register_state(self._eps_max, "eps_max")
        self.register_state(self._eps_min, "eps_min")
        self.register_state(self._eps_decay, "eps_decay")
        self.register_state(self._model, "critic")
        self.register_state(self._target_model, "target_critic")
        self.register_state(self._exploration, "exploration")
        self.register_state(self._step, "step")
        self.set_critic(self._model)
        self.target_critic = self._target_model

        self.target_update()

    def target_update(self):
        self._target_model.load_state_dict(self._model.state_dict())

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
        if behavior == BehaviorMode.EXPLORATION:
            self._step += 1
            if np.random.random() < self._calc_eps():
                actions = self.action_space.n
                # XXX(ming): maybe legal moves move to environment APIs could be a better choice.
                if "legal_moves" in kwargs:
                    actions = kwargs["legal_moves"]
                elif "action_mask" in kwargs:
                    actions = np.where(kwargs["action_mask"] == 1)[0]
                action = np.random.choice(actions)
                action_prob = torch.zeros(self.action_space.n)
                action_prob[action] = 1.0
                return action, action_prob, {Episode.ACTION_DIST: action_prob}
        probs = torch.softmax(self._model(np.asarray([observation])), dim=-1)
        if "legal_moves" in kwargs:
            mask = torch.zeros_like(probs)
            mask[:, kwargs["legal_moves"]] = 1
            probs = mask * probs
        elif "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"]).to(probs.device)
            mask = mask.long().unsqueeze(0)
            probs = mask * probs
        action = probs.argmax().view(1)
        action_prob = torch.zeros_like(probs, device=probs.device)
        action_prob[:, action] = 1.0
        action_prob = action_prob.squeeze(0)
        extra_info = {Episode.ACTION_DIST: action_prob}

        return action.item(), action_prob, extra_info

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        raise NotImplementedError

    def reset_step(self):
        self._step = 0

    def soft_update(self, tau=0.01):
        for target_param, param in zip(
            self._target_model.parameters(), self._model.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
