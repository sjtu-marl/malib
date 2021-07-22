"""
Support discrete action space only
"""

from typing import Any

import gym
import numpy as np
import torch
import torch.nn.functional as F

from malib.algorithm.common.reward import Reward
from malib.utils.typing import (
    DataTransferType,
    Dict,
    ObservationSpaceType,
    Tuple,
    BehaviorMode,
)
from malib.algorithm.common.model import get_model
from malib.utils.preprocessor import get_preprocessor
from malib.algorithm.common import misc
from malib.backend.datapool.offline_dataset_server import Episode


class AdvIRLReward(Reward):
    def __init__(
        self,
        registered_name: str,
        reward_type: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(AdvIRLReward, self).__init__(
            registered_name=registered_name,
            reward_type=reward_type,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )
        assert reward_type in [
            "GAIL",
            "GAIL2",
            "ARIL",
            "FAIRL",
        ], "Adversarial reward type must be in ['GAIL', 'GAIL2', 'ARIL', 'FAIRL'] !"

        # action_dim = get_preprocessor(action_space)(action_space).size
        self._discrete_action = isinstance(action_space, gym.spaces.Discrete)

        self.set_discriminator(
            get_model(self.model_config["discriminator"])(
                observation_space, action_space, self.custom_config["use_cuda"], concat=True,
            )
        )

        self.register_state(self.discriminator, "discriminator")

    def compute_rewards(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> DataTransferType:
        disc_logits = self.discriminator(np.concatenate([observation, action], axis=1))

        if self.reward_type == "airl":
            # If you compute log(D) - log(1-D) then you just get the logits
            reward = disc_logits
        elif self.reward_type == "gail":  # -log (1-D) > 0
            reward = F.softplus(disc_logits, beta=1)  # F.softplus(disc_logits, beta=-1)
        elif self.reward_type == "gail2":  # log D < 0
            reward = F.softplus(
                disc_logits, beta=-1
            )  # F.softplus(disc_logits, beta=-1)
        else:  # fairl
            reward = torch.exp(disc_logits) * (-1.0 * disc_logits)

        return self.clip_rewards(reward)

    def compute_reward(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> Tuple[Any]:

        return self.compute_rewards(observation, action)

    @property
    def discriminator(self) -> Any:
        """ Return reward function, cannot be None """

        return self._discriminator

    def set_discriminator(self, discriminator) -> None:
        """Set discriminator model. Note repeated assign will raise a warning

        :raise RuntimeWarning, repeated assign.
        """

        if self._discriminator is not None:
            raise RuntimeWarning("repeated discriminator assign")
        self._discriminator = discriminator
