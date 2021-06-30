# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import torch
from torch._C import dtype

from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType


class RandomPolicy(Policy):
    def __init__(
        self,
        registered_name,
        observation_space,
        action_space,
        model_config,
        custom_config,
    ):
        super().__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        batch_size = len(observation)
        actions = range(self.action_space.n)
        action_prob = torch.zeros((batch_size, self.action_space.n))
        # if "legal_moves" in kwargs:
        #     actions = kwargs["legal_moves"]
        # if "action_mask" in kwargs:
        #     actions = np.where(kwargs["action_mask"] == 1)[0]
        #     action_prob[actions] = 1.0
        #     action_prob /= action_prob.sum()
        actions = np.random.choice(actions, size=batch_size)
        action_prob.scatter_(
            -1, torch.from_numpy(actions.reshape((batch_size, -1))), 1.0
        )
        return actions, None, {"action_probs": action_prob}

    def compute_action(self, observation: DataTransferType, **kwargs) -> Any:
        actions = range(self.action_space.n)
        action_prob = torch.zeros(self.action_space.n)
        # if "legal_moves" in kwargs:
        #     actions = kwargs["legal_moves"]
        if "action_mask" in kwargs:
            actions = np.where(kwargs["action_mask"] == 1)[0]
            action_prob[actions] = 1.0
            action_prob /= action_prob.sum()
        action = np.random.choice(actions)

        return action, None, {"action_probs": action_prob}

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def set_weights(self, parameters):
        pass
