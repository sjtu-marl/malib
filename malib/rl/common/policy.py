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

from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Tuple, Union
from collections import namedtuple

import copy
import torch
import torch.nn as nn
import gym
import numpy as np

from gym import spaces

from malib.utils.preprocessor import get_preprocessor, Preprocessor
from malib.common.distributions import make_proba_distribution, Distribution
from malib.models.config import ModelConfig
from malib.models.model_client import ModelClient


class SimpleObject:
    def __init__(self, obj, name):
        # assert hasattr(obj, name), f"Object: {obj} has no such attribute named `{name}`"
        self.obj = obj
        self.name = name

    def __str__(self):
        return f"<SimpleObject, name={self.name}, obj={self.obj}>"

    def __repr__(self):
        return f"<SimpleObject, name={self.name}, obj={self.obj}>"

    def load_state_dict(self, v):
        setattr(self.obj, self.name, v)

    def state_dict(self):
        value = getattr(self.obj, self.name)
        return value


Action = np.ndarray
ActionDist = np.ndarray
Logits = np.ndarray

PolicyReturn = namedtuple("PolicyReturn", "action,action_dist,logits,others")


class Policy(metaclass=ABCMeta):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Union[ModelConfig, Dict[str, Any]],
        **kwargs,
    ):
        _locals = locals()
        _locals.pop("self")
        self._init_args = _locals
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config
        self._custom_config = kwargs
        self._preprocessor = get_preprocessor(
            observation_space,
            mode=kwargs.get("preprocess_mode", "flatten"),
        )(observation_space)

        self._device = torch.device(kwargs.get("device", "cpu"))

        if isinstance(action_space, spaces.Discrete):
            self._action_type = "discrete"
        elif isinstance(action_space, spaces.Box):
            self._action_type = "continuous"
        else:
            raise NotImplementedError(
                "Does not support other action space type settings except Box and Discrete. {}".format(
                    type(action_space)
                )
            )

        self._dist_fn: Distribution = make_proba_distribution(
            action_space=action_space,
            use_sde=kwargs.get("use_sde", False),
            dist_kwargs=kwargs.get("dist_kwargs", None),
        )
        self._model = kwargs.get("model_client")
        if self._model is None:
            if kwargs.get("model_entry_point"):
                self._model = ModelClient(
                    kwargs["model_entry_point"],
                    ModelConfig(lambda **x: self.create_model(), model_config),
                )
            else:
                self._model = self.create_model().to(self._device)

    def create_model(self) -> nn.Module:
        raise NotImplementedError

    @property
    def dist_fn(self) -> Distribution:
        return self._dist_fn

    @property
    def action_type(self) -> str:
        return self._action_type

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def model_config(self) -> Dict[str, Any]:
        if isinstance(self._model_config, ModelConfig):
            return self._model_config.to_dict()
        else:
            return copy.deepcopy(self._model_config)

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def custom_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._custom_config)

    def load_state_dict(
        self, state_dict: Dict[str, Any] = None, checkpoint: str = None
    ) -> "Policy":
        """Load state dict outside.

        Args:
            state_dict (Dict[str, Any]): A dict of states.
        """

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        elif checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))

        return self

    def state_dict(
        self, device: Union[torch.DeviceObjType, str] = None
    ) -> Dict[str, Any]:
        """Return state dict of model.

        Args:
            device (Union[torch.DeviceObjType, str], optional): Device name. Defaults to None.

        Returns:
            Dict[str, Any]: A state dict
        """

        if device is None:
            res = self.model.state_dict()
        else:
            res = {}
            for k, v in self.model.state_dict().items():
                res[k] = v.to(device)

        return res

    def get_initial_state(self, batch_size: int = None):
        return None

    @abstractmethod
    def compute_action(
        self,
        observation: torch.Tensor,
        act_mask: Union[torch.Tensor, None],
        evaluate: bool,
        hidden_state: Any = None,
        **kwargs,
    ) -> PolicyReturn:
        pass

    def save(self, path, global_step=0, hard: bool = False):
        state_dict = {"global_step": global_step, **self.state_dict()}
        torch.save(state_dict, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        print(
            f"[Model Loading] Load policy model with global step={state_dict.pop('global_step')}"
        )
        self.load_state_dict(state_dict)

    def reset(self, **kwargs):
        """Reset parameters or behavior policies."""
        pass

    @classmethod
    def copy(cls, instance: "Policy", replacement: Dict) -> "Policy":
        """Self copy, from a given instance. The replacement is a dict of new arguments to override.

        Args:
            instance (Policy): A policy instance to copy from. Must be an instance of cls.
            replacement (Dict): A dict of new arguments to override.

        Returns:
            New policy instance.
        """

        kwargs = {**replacement, **instance._init_args}
        return cls(**kwargs)

    def to(self, device: str = None, use_copy: bool = False) -> "Policy":
        """Convert policy to a given device. If `use_copy`, then return a copy. If device is None, do not change device.

        Args:
            device (str): Device identifier.
            use_copy (bool, optional): User copy or not. Defaults to False.

        Raises:
            NotImplementedError: Not implemented error.

        Returns:
            Policy: A policy instance
        """

        if isinstance(device, str):
            device = torch.device(device)

        if device is None:
            device = (
                torch.device("cpu") if "cuda" not in self.device.type else self.device
            )

        cond1 = "cpu" in device.type and "cuda" in self.device.type
        cond2 = "cuda" in device.type and "cuda" not in self.device.type

        if "cpu" in device.type:
            _device = device
        else:
            _device = self.device

        replacement = {}
        if cond1 or cond2:
            _model = self.model.to(device)
            if not use_copy:
                setattr(self, "model", _model)
            else:
                replacement["model_client"] = _model
        else:
            replacement["model_client"] = self.model

        if use_copy:
            ret = self.copy(self, replacement=replacement)
        else:
            ret = self

        return ret

    def parameters(self, recurse: bool = True):
        """Returns an iterator over module parameters.
        This is typically passed to an optimizer.

        Args:
            recurse (bool, optional): If True, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module. Defaults to True.

        Yields:
            Parameter: module parameter
        """

        return self.model.parameters(recurse=recurse)

    def coordinate(self, state: Dict[str, torch.Tensor], message: Any) -> Any:
        """Coordinate with other agents here"""

        pass
