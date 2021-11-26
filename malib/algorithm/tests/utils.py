import gym

from malib.utils.typing import Dict, Any
from malib.algorithm.random.policy import RandomPolicy


def build_random_policy(
    obs_space: gym.Space, act_space: gym.Space, custom_config: Dict[str, Any] = None
):
    return RandomPolicy(
        registered_name="random",
        observation_space=obs_space,
        action_space=act_space,
        custom_config=custom_config,
        model_config=None,
    )


def build_random_nn_policy(
    obs_space: gym.Space,
    act_space: gym.Space,
    nn_type: str,
    custom_config: Dict[str, Any] = None,
):
    model_config = None

    if nn_type == "fc":
        raise NotImplementedError
    elif nn_type == "vision":
        raise NotImplementedError
    elif nn_type == "rnn":
        raise NotImplementedError
    elif nn_type == "lstm":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return RandomPolicy(
        registered_name="random",
        observation_space=obs_space,
        action_space=act_space,
        custom_config=custom_config,
        model_config=model_config,
    )
