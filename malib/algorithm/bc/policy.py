from typing import Any, Dict

import gym
import torch

from torch.distributions.categorical import Categorical

from malib.algorithm.common.model import MLPActor
from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType


class BehaviorCloning(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(BehaviorCloning, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.lr = custom_config.get("lr", 1e-4)
        self.gamma = custom_config.get("gamma", 0.98)
        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)
        # FIXME(ming): limit action space types
        self.action_dim = (
            action_space.n if self.discrete_action else action_space.shape[0]
        )

        self._actor = MLPActor(
            obs_dim=self.preprocessor.size,
            act_dim=self.action_dim,
            model_config=model_config,
        )

        self.register_state(self._actor, "actor")

    def actor(self) -> Any:
        return self._actor

    def critic(self) -> Any:
        return None

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        """ Compute actions for a batched observations in optimization stage """

        if self.discrete_action:
            probs = self._actor(observation)
            m = Categorical(probs=probs)
            actions = m.sample()
        else:
            actions = self._actor(observation)

        return actions.cpu().numpy()

    def compute_action(self, observation: DataTransferType, **kwargs) -> Any:
        """ Compute action for a piece of observation """

        if self.discrete_action:
            probs = self._actor(observation).detach()
            mask = None
            if "action_mask" in kwargs:
                mask = torch.FloatTensor(kwargs["action_mask"]).to(probs.device)
                mask = mask.long().unsqueeze(0)
                probs = mask * probs
            action = probs.argmax().view(1)
            extra_info = {}
            action_prob = torch.zeros_like(probs, device=probs.device)
            if mask is not None:
                active_indices = mask > 0
                action_prob[active_indices] = probs[active_indices] / probs.sum()
            extra_info["action_probs"] = action_prob.squeeze(0)
        else:
            action = self._actor(observation).detach()
            extra_info = None

        return action.item(), None, extra_info
