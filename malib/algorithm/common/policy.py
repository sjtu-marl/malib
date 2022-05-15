"""
Implementation of basic PyTorch-based policy class
"""

import gym

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from malib.utils import errors
from malib.utils.typing import (
    DataTransferType,
    ModelConfig,
    Dict,
    Any,
    Tuple,
    Callable,
    List,
)
from malib.utils.preprocessor import get_preprocessor, Mode
from malib.utils.notations import deprecated
from malib.algorithm.common.model import Model


class SimpleObject:
    def __init__(self, obj, name):
        assert hasattr(obj, name), f"Object: {obj} has no such attribute named `{name}`"
        self.obj = obj
        self.name = name

    def load_state_dict(self, v):
        setattr(self.obj, self.name, v)

    def state_dict(self):
        value = getattr(self.obj, self.name)
        return value


DEFAULT_MODEL_CONFIG = {
    "actor": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
    "critic": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
}


class Policy(metaclass=ABCMeta):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: ModelConfig = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """Create a policy instance.

        :param str registered_name: Registered policy name.
        :param gym.spaces.Space observation_space: Raw observation space of related environment agent(s), determines
            the model input space.
        :param gym.spaces.Space action_space: Raw action space of related environment agent(s).
        :param Dict[str,Any] model_config: Model configuration to construct models. Default to None.
        :param Dict[str,Any] custom_config: Custom configuration, includes some hyper-parameters. Default to None.
        """

        self.registered_name: str = registered_name
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space
        self.device = torch.device("cpu")

        self.custom_config = {
            "gamma": 0.99,
            "use_cuda": False,
            "use_dueling": False,
            "preprocess_mode": Mode.FLATTEN,
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
        self._actor = None
        self._critic = None
        self._exploration_callback = None
        self._kwargs = kwargs

    @property
    def exploration_callback(self) -> Callable:
        return self._exploration_callback

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
        """Return a dict of basic attributes to identify policy.

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

    @abstractmethod
    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        """Compute batched actions for the current policy with given inputs.

        Legal keys in kwargs:

        - behavior_mode: behavior mode used to distinguish different behavior of compute actions.
        - action_mask: action mask.
        """

    @abstractmethod
    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[DataTransferType, DataTransferType, List[DataTransferType]]:
        """Compute single action when rollout at each step, return 3 elements:
        action, action_dist, a list of rnn_state
        """

    def get_initial_state(self, batch_size: int = None) -> List[DataTransferType]:
        """Return a list of rnn states if models are rnns"""

        return []

    def state_dict(self, device="cpu"):
        """Return state dict in real time"""

        res = {}
        for k, v in self._state_handler_dict.items():
            if isinstance(v, Model) and "cpu" in device:
                res[k] = {_k: _v.cpu() for _k, _v in v.state_dict().items()}
            else:
                res[k] = v.state_dict()
        return res

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict outside.

        Note that the keys in `state_dict` should be existed in state handler.

        :param state_dict: Dict[str, Any], A dict of state dict
        :raise: KeyError
        """

        for k, v in state_dict.items():
            self._state_handler_dict[k].load_state_dict(v)

    # XXX(ziyu): Add tests for it.
    def set_weights(self, parameters: Dict[str, Any]):
        """Set parameter weights.

        :param parameters: Dict[str, Any], A dict of parameters.
        :return:
        """

        for k, v in parameters.items():
            # FIXME(ming): strict mode for parameter reload
            self._state_handler_dict[k].load_state_dict(v)

    def set_actor(self, actor) -> None:
        """Set actor. Note repeated assign will raise a warning

        :raise RuntimeWarning, repeated assign.
        """

        # if self._actor is not None:
        #     raise RuntimeWarning("repeated actor assign")
        self._actor = actor

    def set_critic(self, critic):
        """Set critic"""

        # if self._critic is not None:
        #     raise RuntimeWarning("repeated critic assign")
        self._critic = critic

    @property
    def actor(self) -> Any:
        """Return policy, cannot be None"""

        return self._actor

    @property
    def critic(self) -> Any:
        """Return critic, can be None"""

        return self._critic

    @deprecated
    def train(self):
        pass

    @deprecated
    def eval(self):
        pass

    # @abstractmethod
    def reset(self):
        """Reset policy intermediates"""
        pass

    def to_device(self, device):
        self.device = device
        return self

    def value_function(self, *args, **kwargs):
        """Compute values of critic."""
        raise NotImplementedError
