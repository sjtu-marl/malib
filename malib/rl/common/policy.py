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
from enum import IntEnum

import torch
import torch.nn as nn

from gym import spaces

from malib.utils.preprocessor import get_preprocessor
from malib.common.distributions import make_proba_distribution, Distribution


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


Action = Any
ActionDist = Any
Logits = Any


class Policy(metaclass=ABCMeta):
    def __init__(
        self, observation_space, action_space, model_config, custom_config, **kwargs
    ):
        _locals = locals()
        _locals.pop("self")
        self._init_args = _locals
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config or {}
        self._custom_config = custom_config or {}
        self._state_handler_dict = {}
        self._preprocessor = get_preprocessor(
            observation_space,
            mode=self._custom_config.get("preprocess_mode", "flatten"),
        )(observation_space)

        self._device = torch.device(
            "cuda" if self._custom_config.get("use_cuda") else "cpu"
        )

        self._registered_networks: Dict[str, nn.Module] = {}

        if isinstance(action_space, spaces.Discrete):
            self.action_type = "discrete"
        elif isinstance(action_space, spaces.Box):
            self.action_type = "continuous"
        else:
            raise NotImplementedError(
                "Does not support other action space type settings except Box and Discrete. {}".format(
                    type(action_space)
                )
            )

        self.use_cuda = self._custom_config.get("use_cuda", False)
        self.dist_fn: Distribution = make_proba_distribution(
            action_space=action_space,
            use_sde=custom_config.get("use_sde", False),
            dist_kwargs=custom_config.get("dist_kwargs", None),
        )

    @property
    def model_config(self):
        return self._model_config

    @property
    def device(self) -> str:
        return self._device

    @property
    def custom_config(self) -> Dict[str, Any]:
        return self._custom_config

    @property
    def target_actor(self):
        return self._target_actor

    @target_actor.setter
    def target_actor(self, value: Any):
        self._target_actor = value

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, value: Any):
        self._actor = value

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, value: Any):
        self._critic = value

    @property
    def target_critic(self):
        return self._target_critic

    @target_critic.setter
    def target_critic(self, value: Any):
        self._target_critic = value

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict outside.

        Args:
            state_dict (Dict[str, Any]): A dict of states.
        """

        for k, v in state_dict.items():
            self._state_handler_dict[k].load_state_dict(v)

    def state_dict(self, device=None):
        """Return state dict in real time"""

        if device is None:
            res = {k: v.state_dict() for k, v in self._state_handler_dict.items()}
        else:
            res = {}
            for k, v in self._state_handler_dict.items():
                if isinstance(v, torch.nn.Module):
                    tmp = {}
                    for _k, _v in v.state_dict().items():
                        tmp[_k] = _v.cpu()
                else:
                    tmp = v.state_dict()
                res[k] = tmp
        return res

    def register_state(self, obj: Any, name: str) -> None:
        """Register state of obj. Called in init function to register model states.

        Example:
            >>> class CustomPolicy(Policy):
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

        Args:
            obj (Any): Any object, for non `torch.nn.Module`, it will be wrapped as a `Simpleobject`.
            name (str): Humanreadable name, to identify states.

        Raises:
            errors.RepeatedAssignError: [description]
        """

        # if not isinstance(obj, nn.Module):
        if obj.__class__.__module__ == "builtins":
            n = SimpleObject(self, name)
            n.load_state_dict(obj)
            obj = n

        self._state_handler_dict[name] = obj
        if isinstance(obj, nn.Module):
            self._registered_networks[name] = obj

    def deregister_state(self, name: str):
        if self._state_handler_dict.get(name) is None:
            print(f"No such state tagged with: {name}")
        else:
            self._state_handler_dict.pop(name)
            print(f"Deregister state tagged with: {name}")

    def get_initial_state(self, batch_size: int = None):
        return None

    @property
    def preprocessor(self):
        return self._preprocessor

    @abstractmethod
    def compute_action(
        self,
        observation: torch.Tensor,
        act_mask: Union[torch.Tensor, None],
        evaluate: bool,
        hidden_state: Any = None,
        **kwargs,
    ) -> Tuple[Action, ActionDist, Logits, Any]:
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
    def copy(cls, instance, replacement: Dict):
        return cls(replacement=replacement, **instance._init_args)

    @property
    def registered_networks(self) -> Dict[str, nn.Module]:
        return self._registered_networks

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

        if device is None:
            device = "cpu" if not self.use_cuda else "cuda"

        cond1 = "cpu" in device and self.use_cuda
        cond2 = "cuda" in device and not self.use_cuda

        if "cpu" in device:
            use_cuda = False
        else:
            use_cuda = self._custom_config.get("use_cuda", False)

        replacement = {}
        if cond1 or cond2:
            # retrieve networks here
            for k, v in self.registered_networks.items():
                _v = v.to(device)
                if not use_copy:
                    setattr(self, k, _v)
                else:
                    replacement[k] = _v
        else:
            # fixed bug: replacement cannot be None.
            for k, v in self.registered_networks.items():
                replacement[k] = v

        if use_copy:
            ret = self.copy(self, replacement=replacement)
        else:
            self.use_cuda = use_cuda
            ret = self

        return ret

    def parameters(self) -> Dict[str, Dict]:
        """Return trainable parameters."""

        res = {}
        for name, net in self.registered_networks.items():
            res[name] = net.parameters()
        return res

    def update_parameters(self, parameter_dict: Dict[str, Any]):
        """Update local parameters with given parameter dict.

        Args:
            parameter_dict (Dict[str, Parameter]): A dict of paramters
        """

        for k, parameters in parameter_dict.items():
            target = self.registered_networks[k]
            for target_param, param in zip(target.parameters(), parameters):
                target_param.data.copy_(param.data)

    def coordinate(self, state: Dict[str, torch.Tensor], message: Any) -> Any:
        """Coordinate with other agents here"""

        raise NotImplementedError
