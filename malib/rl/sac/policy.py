from typing import Any, Dict, Tuple

import gym
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.distributions import Normal

from malib.utils.typing import DataTransferType, BehaviorMode
from malib.rl.common.policy import Policy
from malib.rl.common import misc


class SAC(Policy):
    def __init__(
        self, observation_space, action_space, model_config, custom_config, **kwargs
    ):
        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )
