"""
Implementation of basic PyTorch-based policy class
"""

import gym

from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn

from malib.utils import errors
from malib.utils.typing import (
    List,
    DataTransferType,
    ModelConfig,
    Dict,
    Any,
    Tuple,
    Callable,
    BehaviorMode,
    Sequence,
    Union,
)
from malib.utils.preprocessor import get_preprocessor, Mode
from malib.utils.notations import deprecated
from malib.envs.tabular.game import Game as TabularGame
from malib.envs.tabular.state import State as TabularGameState


class SimpleObject:
    def __init__(self, obj, value):
        self.obj = obj
        self.value = value

    def __getstate__(self):
        return {"value": self.value}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def load_state_dict(self, v):
        self.value = v

    def state_dict(self):
        return self.value


DEFAULT_MODEL_CONFIG = {
    "actor": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": "Softmax"},
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


class TabularPolicy:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        callable_policy: callable,
        batched_callable_policy: callable,
        value_func: callable = None,
        values_func: callable = None,
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self._game_states = []
        self._tabular_dist = None

        assert isinstance(
            action_space, gym.spaces.Discrete
        ), f"The action space is illegal, expected should be `gym.spaces.Discrete`, while received {type(action_space)}"
        assert isinstance(
            observation_space, gym.spaces.Discrete
        ), f"The observation space is illegal, expected should be `gym.spaces.Discrete`, while received {type(observation_space)}"

        self._callable_policy = callable_policy
        self._batched_callable_policy = batched_callable_policy

        self._value_func = value_func
        self._values_func = values_func

    @property
    def action_probability_array(self) -> Dict[TabularGameState, np.ndarray]:
        assert self._tabular_dist is not None
        return self._tabular_dist

    def set_states_to_init_policy(self, states: Sequence[TabularGameState]):
        self._game_states = states
        # init policy in tabular
        self._tabular_dist = {
            state: np.zeros(self._action_space.n) for state in self._game_states
        }

    def action_probability(self, state: TabularGameState):
        return self._callable_policy(state)

    def action_probabilities(self, states: Sequence[TabularGameState]):
        return self._batched_callable_policy(states)

    def value(self, state: TabularGameState, walk: bool = True):
        if not walk:
            assert self._value_func is not None and callable(self._value_func), type(
                self._value_func
            )
            return self._value_func(state)
        else:
            # TODO(ming): recursively walk through the game tree to estimate the state value
            raise NotImplementedError

    def values(self, states: Sequence[TabularGameState], walk: bool = True):
        if not walk:
            assert self._values_func is not None and callable(self._values_func), type(
                self._values_func
            )
            return self._values_func(states)
        else:
            # TODO(ming): recursively walk through the game tree to estimate the state vlaues
            raise NotImplementedError

    def __call__(self, state: Union[TabularGameState, Sequence[TabularGameState]]):
        """Turns the policy into a callable.

        :param Any state: The current state used to compute strategy
        :return: A `dict` of {action: probability}` for the given state
        """
        return self.action_probability(state)


class Policy(metaclass=ABCMeta):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: ModelConfig = None,
        custom_config: Dict[str, Any] = None,
    ):
        """Create a policy instance.

        :param str registered_name: Registered policy name.
        :param gym.spaces.Space observation_space: Raw observation space of related environment agent(s), determines
            the model input space.
        :param gym.spaces.Space action_space: Raw action space of related environment agent(s).
        :param Dict[str,Any] model_config: Model configuration to construct models. Default to None.
        :param Dict[str,Any] custom_config: Custom configuration, includes some hyper-parameters. Default to None.
        """

        self.registered_name = registered_name
        self.observation_space = observation_space
        self.action_space = action_space

        self.custom_config = {
            "gamma": 0.98,
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
            obj = SimpleObject(self, obj)
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

        pass

    @abstractmethod
    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[Any, Any, Any]:
        """Compute single action when rollout at each step, return 3 elements:
        action, None, extra_info['actions_prob']
        """

        pass

    def state_dict(self):
        """ Return state dict in real time """

        return {k: v.state_dict() for k, v in self._state_handler_dict.items()}

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
        """ Return policy, cannot be None """

        return self._actor

    @property
    def critic(self) -> Any:
        """ Return critic, can be None """

        return self._critic

    @deprecated
    def train(self):
        pass

    @deprecated
    def eval(self):
        pass

    def to_tabular(self) -> TabularPolicy:
        """Convert RL policy to tabular policy."""

        def _callable_policy(state: TabularGameState):
            valid_action_index_with_mask = state.legal_actions_mask()
            info_state_vector = state.information_state_tensor()
            obs = self.preprocessor.transform(
                {
                    "observation": np.asarray(info_state_vector),
                    "action_mask": np.asarray(valid_action_index_with_mask),
                }
            )
            _, action_probs, _ = self.compute_action(
                observation=obs, behavior_mode=BehaviorMode.EXPLOITATION
            )
            return {
                state.actions[idx]: action_probs[idx]
                for idx, v in enumerate(valid_action_index_with_mask)
                if v == 1.0
            }

        def _batched_callable_policy(states: Sequence[TabularGameState]):
            obs = np.asarray(
                [
                    self.preprocessor.transform(
                        {
                            "observation": np.asarray(state.information_state_tensor()),
                            "action_mask": np.asarray(state.legal_actions_mask()),
                        }
                    )
                    for state in states
                ]
            )

            _, batched_action_probs, _ = self.compute_actions(
                observation=obs, behavior_mode=BehaviorMode.EXPLOITATION
            )
            batched_action_probs = batched_action_probs.tolist()
            actions = [
                {
                    state.actions[idx]: action_probs[idx]
                    for idx, v in enumerate(state.legal_actions_mask())
                    if v == 1.0
                }
                for state, action_probs in zip(states, batched_action_probs)
            ]
            return actions

        tabular_policy = TabularPolicy(
            self.observation_space,
            self.action_space,
            _callable_policy,
            _batched_callable_policy,
        )
        return tabular_policy
