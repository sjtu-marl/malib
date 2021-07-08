import numpy as np
import pytest
import gym
import logging

from gym.spaces import Discrete, Box
from malib.utils.typing import BehaviorMode, Dict, Any
from malib.algorithm.dqn import DQN, DQNLoss, DQNTrainer


@pytest.fixture(scope="session")
def agents():
    return {f"AGENT-{i}" for i in range(1)}


@pytest.fixture(scope="session")
def action_spaces(agents) -> Dict[str, gym.Space]:
    return {agent: Discrete(5) for agent in agents}


@pytest.fixture(scope="session")
def default_network_config() -> Dict[str, Any]:
    raise NotImplementedError


@pytest.fixture(scope="session")
def one_dim_observation_spaces(agents) -> Dict[str, gym.Space]:
    return {agent: Box(low=0.0, high=1.0, shape=(3,)) for agent in agents}


@pytest.fixture(scope="session")
def two_dim_observation_spaces(agents) -> Dict[str, gym.Space]:
    return {agent: Box(low=0.0, high=1.0, shape=(2, 3)) for agent in agents}


@pytest.fixture(scope="session")
def default_model_config():
    return {}


@pytest.fixture(scope="session")
def custom_config():
    return {}


def test_low_rank_obs_action_dim(
    agents,
    one_dim_observation_spaces,
    action_spaces,
    default_model_config,
    custom_config,
):
    # model sharing
    obs_space = list(one_dim_observation_spaces.values())[0]
    act_space = list(action_spaces.values())[0]

    dqn = DQN(
        "dqn",
        observation_space=obs_space,
        action_space=act_space,
        model_config=default_model_config,
        custom_config=custom_config,
    )

    one_slice_obs = dqn.observation_space.sample()
    _repeats = tuple([1 for _ in one_slice_obs.shape])
    _repeats = (3,) + _repeats
    logging.debug(f"repeats and shape is: {_repeats} {one_slice_obs.shape}")
    multi_slice_obs = np.tile(one_slice_obs, _repeats)

    action, action_prob, _ = dqn.compute_action(
        [one_slice_obs], behavior_mode=BehaviorMode.EXPLOITATION
    )
    assert action.shape == (1, 1), action.shape
    action, action_prob, _ = dqn.compute_action(
        multi_slice_obs, behavior_mode=BehaviorMode.EXPLOITATION
    )
    assert action.shape == (_repeats[0], 1), action.shape
    action, action_prob, _ = dqn.compute_action(
        one_slice_obs, behavior_mode=BehaviorMode.EXPLORATION
    )
    assert action.shape == (1, 1), action.shape
    action, action_prob, _ = dqn.compute_action(
        multi_slice_obs, behavior_mode=BehaviorMode.EXPLORATION
    )
    assert action.shape == (_repeats[0], 1), action.shape


def test_high_rank_obs_action_dim(agents, two_dim_observation_spaces, action_spaces):
    pass
