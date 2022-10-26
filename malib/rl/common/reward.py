# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, Any, Tuple, Callable
from abc import ABCMeta, abstractmethod

import gym
import torch.nn as nn

from malib.utils import errors
from malib.utils.typing import DataTransferType
from malib.utils.preprocessor import get_preprocessor, Mode
from malib.utils.notations import deprecated
from malib.rl.common.policy import SimpleObject

import torch

DEFAULT_MODEL_CONFIG = {
    "reward": {
        "network": "mlp",
        "layers": [
            {"units": 128, "activation": "ReLU"},
            {"units": 128, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
}


class Reward(metaclass=ABCMeta):
    def __init__(
        self,
        registered_name: str,
        reward_type: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        """Create a reward model instance.

        :param str registered_name: Registered policy name.
        :param str reward_type: Reward function deteiled type in practice.
        :param gym.spaces.Space observation_space: Raw observation space of related environment agent(s), determines
            the model input space.
        :param gym.spaces.Space action_space: Raw action space of related environment agent(s).
        :param Dict[str,Any] model_config: Model configuration to construct models. Default to None.
        :param Dict[str,Any] custom_config: Custom configuration, includes some hyper-parameters. Default to None.
        """

        self.registered_name = registered_name
        self.reward_type = reward_type
        self.observation_space = observation_space
        self.action_space = action_space

        self.custom_config = {
            "gamma": 0.98,
            "use_cuda": False,
            "use_dueling": False,
            "preprocess_mode": Mode.FLATTEN,
            "clip_max_rews": False,
            "clip_min_rews": False,
            "rew_clip_max": 10,
            "rew_clip_min": -10,
        }
        self.model_config = DEFAULT_MODEL_CONFIG

        if custom_config is None:
            custom_config = {}
        self.custom_config.update(custom_config)

        # FIXME(ming): use deep update rule
        if model_config is None:
            model_config = {}
        self.model_config.update(model_config)

        self.preprocessor = get_preprocessor(
            observation_space, self.custom_config["preprocess_mode"]
        )(observation_space)

        self._state_handler_dict = {}
        self._discriminator = None

    @property
    def exploration_callback(self) -> Callable:
        return self._exploration_callback

    def register_state(self, obj: Any, name: str) -> None:
        """Register state of obj. Called in init function to register model states.

        Example:
            >>> class CustomReward(Reward):
            ...     def __init__(
            ...         self,
            ...         registered_name,
            ...         observation_space,
            ...         action_space,
            ...         model_config,
            ...         custom_config
            ...     ):
            ...     # ...
            ...     actor = MLP(...)
            ...     self.register_state(actor, "actor")

        :param Any obj: Any object, for non `torch.nn.Module`, it will be wrapped as a `Simpleobject`.
        :param str name: Humanreadable name, to identify states.
        :raise: malib.utils.errors.RepeatedAssign
        :return: None
        """

        if not isinstance(obj, nn.Module):
            obj = SimpleObject(self, name)
        if self._state_handler_dict.get(name, None) is not None:
            raise errors.RepeatedAssignError(
                f"state handler named with {name} is not None."
            )
        self._state_handler_dict[name] = obj

    def deregister_state(self, name: str):
        if self._state_handler_dict.get(name) is None:
            print(f"No such state tagged with: {name}")
        else:
            self._state_handler_dict.pop(name)
            print(f"Deregister state tagged with: {name}")

    @property
    def description(self):
        """Return a dict of basic attributes to identify reward.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }

    def clip_rewards(self, rewards):
        if self.custom_config["clip_max_rews"]:
            rewards = torch.clamp(rewards, max=self.rew_clip_max)
        if self.custom_config["clip_min_rews"]:
            rewards = torch.clamp(rewards, min=self.rew_clip_min)

        return rewards

    @abstractmethod
    def compute_rewards(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> DataTransferType:
        """Compute batched rewards for the current policy with given inputs.

        Legal keys in kwargs:

        - behavior_mode: behavior mode used to distinguish different behavior of compute actions.
        - action_mask: action mask.
        """

        pass

    @abstractmethod
    def compute_reward(
        self, observation: DataTransferType, action: DataTransferType, **kwargs
    ) -> Tuple[Any]:
        """Compute single reward when rollout at each step, return 1 elements:
        reward
        """

        pass

    def state_dict(self):
        """Return state dict in real time"""

        res = {k: v.state_dict() for k, v in self._state_handler_dict.items()}
        return res

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict outside.

        Note that the keys in `state_dict` should be existed in state handler.

        :param state_dict: Dict[str, Any], A dict of state dict
        :raise: KeyError
        """

        for k, v in state_dict.items():
            self._state_handler_dict[k].load_state_dict(v)

    def set_weights(self, parameters: Dict[str, Any]):
        """Set parameter weights.

        :param parameters: Dict[str, Any], A dict of parameters.
        :return:
        """

        for k, v in parameters.items():
            # FIXME(ming): strict mode for parameter reload
            self._state_handler_dict[k].load_state_dict(v)

    @deprecated
    def train(self):
        pass

    @deprecated
    def eval(self):
        pass
