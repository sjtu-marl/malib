"""
Environment/VecEnvironment -> Episode (stacked or not) -> postprocessor -> offline dataset
"""
import enum
from typing import List
import numpy as np
import scipy.signal

from malib.utils.typing import AgentID, Dict, PolicyID, Union, Callable
from malib.utils.episode import Episode, EpisodeKey
from malib.algorithm.common.policy import Policy


class PostProcessorType(enum.IntEnum):
    ADVANTAGE = 0
    GAE = 1
    ACCUMULATED_REWORD = 2


# FIXME(ziyu): For loop for episodes at the beginning
def compute_acc_reward(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    # create new placeholder
    for episode in episodes:
        episode[EpisodeKey.ACC_REWARD] = {}

        for aid, data in episode[EpisodeKey.REWARD].items():
            gamma = policy_dict[aid].custom_config["gamma"]
            assert isinstance(
                data, np.ndarray
            ), "Reward must be an numpy array: {}".format(data)
            assert (
                len(data.shape) == 1
            ), "Reward should be a scalar at eatch time step: {}".format(data)
            acc_reward = scipy.signal.lfilter(
                [1], [1, float(-gamma)], data[::-1], axis=0
            )[::-1]
            episode[EpisodeKey.ACC_REWARD][aid] = acc_reward

    return episodes


def compute_advantage(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
    last_r: Dict[AgentID, float],
    use_gae: bool = False,
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    episodes = compute_acc_reward(episodes, policy_dict)
    for episode in episodes:
        episode[EpisodeKey.ADVANTAGE] = {}
        for aid, policy in policy_dict.items():
            use_gae = policy.custom_config.get("use_gae", False)
            use_critic = policy.custom_config.get("use_critic", False)

            if use_gae:
                gamma = policy.custom_config["gamma"]
                v = np.concatenate(
                    [episode[EpisodeKey.STATE_VALUE], np.array([last_r[aid]])]
                )
                delta_t = episode[EpisodeKey.REWARD] + gamma * v[1:] - v[:-1]
                episode[EpisodeKey.ADVANTAGE][aid] = scipy.signal.lfilter(
                    [1], [1, float(-gamma)], delta_t[::-1], axis=0
                )[::-1]
                episode[EpisodeKey.STATE_VALUE_TARGET][aid] = (
                    episode[EpisodeKey.ADVANTAGE][aid]
                    + episode[EpisodeKey.STATE_VALUE][aid]
                )
            else:
                v = np.concatenate(
                    [episode[EpisodeKey.REWARD][aid], np.array([last_r[aid]])]
                )
                acc_r = episode[EpisodeKey.ACC_REWARD][aid]
                if use_critic:
                    episode[EpisodeKey.ADVANTAGE][aid] = (
                        acc_r - episode[EpisodeKey.STATE_VALUE][aid]
                    )
                    episode[EpisodeKey.STATE_VALUE_TARGET][aid] = episode[
                        EpisodeKey.ACC_REWARD
                    ][aid].copy()
                else:
                    episode[EpisodeKey.ADVANTAGE][aid] = episode[EpisodeKey.ACC_REWARD][
                        aid
                    ]
                    episode[EpisodeKey.STATE_VALUE_TARGET][aid] = np.zeros_like(
                        episode[EpisodeKey.ADVANTAGE][aid]
                    )
    return episodes


def compute_gae(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    last_r = {}
    for episode in episodes:
        for aid, dones in episode[EpisodeKey.DONE].items():
            if dones[-1]:
                last_r[aid] = 0.0
            else:
                # compute value as last r
                assert hasattr(policy_dict[aid], "value_functon")
                last_r[aid] = policy_dict[aid].value_function(episodes, agent_key=aid)

    episode = compute_advantage(episodes, policy_dict, last_r=last_r, use_gae=True)
    return episode


def compute_value(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
):
    for episode in episodes:
        for aid, policy in policy_dict.items():
            episode[aid][EpisodeKey.STATE_VALUE] = policy.value_function(**episode[aid])
    return episodes


# XXX(ming): require test
def copy_next_frame(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
):
    for episode in episodes:
        for aid in policy_dict:
            agent_episode = episode[aid]
            agent_episode[EpisodeKey.NEXT_OBS] = agent_episode[
                EpisodeKey.CUR_OBS
            ].copy()
            if EpisodeKey.CUR_STATE in agent_episode:
                agent_episode[EpisodeKey.NEXT_STATE] = agent_episode[
                    EpisodeKey.CUR_STATE
                ].copy()
    return episodes


def default_processor(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    return episodes


def get_postprocessor(
    processor_types: List[
        Union[str, Callable[[Episode, Dict[AgentID, Policy]], Episode]]
    ]
) -> Callable[[Episode, Dict[AgentID, Policy]], Episode]:
    for processor_type in processor_types:
        if callable(processor_type):
            yield processor_type
        # XXX(ming): we will allow heterogeneous processor settings
        elif processor_type == "gae":
            yield compute_gae
        elif processor_type == "acc_reward":
            yield compute_acc_reward
        elif processor_type == "advantage":
            yield compute_advantage
        elif processor_type == "default":
            yield default_processor
        elif processor_type == "value":
            yield compute_value
        elif processor_type == "copy_next_frame":
            yield copy_next_frame
        else:
            return ValueError("Disallowed processor type: {}".format(processor_type))
