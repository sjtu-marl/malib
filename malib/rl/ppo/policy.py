from typing import Tuple, Dict, Any, List
import traceback
import gym
import torch
import numpy as np
import copy

from functools import reduce
from operator import mul

from torch.nn import functional as F
from torch.distributions import Categorical, Normal

from malib.utils.typing import BehaviorMode, DataTransferType
from malib.utils.episode import Episode
from malib.rl.common.policy import Policy
from malib.rl.common import misc


class PPO(Policy):
    def __init__(
        self, observation_space, action_space, model_config, custom_config, **kwargs
    ):
        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )
