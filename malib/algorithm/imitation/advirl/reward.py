"""
Support discrete action space only
"""

from typing import Any

import gym
import torch

from malib.algorithm.common.reward import Reward
from malib.utils.typing import DataTransferType, Dict, Tuple, BehaviorMode
from malib.algorithm.common.model import get_model
from malib.utils.preprocessor import get_preprocessor
from malib.algorithm.common import misc
from malib.backend.datapool.offline_dataset_server import Episode


class Discriminator(Reward):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(Discriminator, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        action_dim = get_preprocessor(action_space)(action_space).size

        self._discrete_action = isinstance(action_space, gym.spaces.Discrete)
        if not self._discrete_action:
            self._exploration_callback = misc.OUNoise(action_dim)
        else:
            self._exploration_callback = misc.EPSGreedy(action_dim, threshold=0.3)

        self.set_reward_func(
            get_model(self.model_config.get("reward_func"))(
                observation_space, action_space, self.custom_config["use_cuda"]
            )
        )

        self.register_state(self.reward_func, "reward_func")

        self.update_target()

    def compute_rewards(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> DataTransferType:
        pass
        # TODO
        # return pi

    def compute_reward(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> Tuple[Any]:
        pass
        # TODO
        # return act.numpy(), pi.numpy(), {Episode.ACTION_DIST: pi.numpy()}
