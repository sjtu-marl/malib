from typing import Any, Dict, Tuple

import gym
import torch
import numpy as np

from torch.distributions import Categorical, Normal

from malib.utils.episode import Episode
from malib.utils.typing import DataTransferType, BehaviorMode
from malib.rl.common import misc
from malib.rl.common.policy import Policy


class DiscreteSAC(Policy):
    def __init__(
        self, observation_space, action_space, model_config, custom_config, **kwargs
    ):
        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )
