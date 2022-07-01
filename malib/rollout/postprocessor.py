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

import enum
from typing import List, Dict, Union, Callable
import numpy as np
import scipy.signal

from malib.utils.typing import AgentID, PolicyID
from malib.utils.episode import Episode, Episode
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
        # episode[Episode.ACC_REWARD] = {}

        for aid, data in episode.items():
            gamma = policy_dict[aid].custom_config["gamma"]
            assert isinstance(
                data[Episode.REWARD], np.ndarray
            ), "Reward must be an numpy array: {}".format(data[Episode.REWARD])
            assert (
                len(data[Episode.REWARD].shape) == 1
            ), "Reward should be a scalar at eatch time step: {}".format(
                data[Episode.REWARD]
            )
            acc_reward = scipy.signal.lfilter(
                [1], [1, float(-gamma)], data[Episode.REWARD][::-1], axis=0
            )[::-1]
            data[Episode.ACC_REWARD] = acc_reward

    return episodes


def compute_advantage(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
    last_r: Dict[AgentID, float],
    use_gae: bool = False,
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    episodes = compute_acc_reward(episodes, policy_dict)
    for agent_episode in episodes:
        for aid, policy in policy_dict.items():
            episode = agent_episode[aid]
            use_gae = policy.custom_config.get("use_gae", False)
            use_critic = policy.custom_config.get("use_critic", False)

            if use_gae:
                gamma = policy.custom_config["gamma"]
                v = np.concatenate(
                    [episode[Episode.STATE_VALUE], np.array([last_r[aid]])]
                )
                delta_t = episode[Episode.REWARD] + gamma * v[1:] - v[:-1]
                episode[Episode.ADVANTAGE] = scipy.signal.lfilter(
                    [1], [1, float(-gamma)], delta_t[::-1], axis=0
                )[::-1]
                episode[Episode.STATE_VALUE_TARGET] = (
                    episode[Episode.ADVANTAGE] + episode[Episode.STATE_VALUE]
                )
            else:
                v = np.concatenate([episode[Episode.REWARD], np.array([last_r[aid]])])
                acc_r = episode[Episode.ACC_REWARD]
                if use_critic:
                    episode[Episode.ADVANTAGE] = acc_r - episode[Episode.STATE_VALUE]
                    episode[Episode.STATE_VALUE_TARGET] = episode[
                        Episode.ACC_REWARD
                    ].copy()
                else:
                    episode[Episode.ADVANTAGE] = episode[Episode.ACC_REWARD]
                    episode[Episode.STATE_VALUE_TARGET] = np.zeros_like(
                        episode[Episode.ADVANTAGE]
                    )
    return episodes


def compute_gae(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
) -> Dict[str, Dict[AgentID, np.ndarray]]:
    last_r = {}
    for agent_episode in episodes:
        for aid, episode in agent_episode.items():
            dones = episode[Episode.DONE]
            if dones[-1]:
                last_r[aid] = 0.0
            else:
                # compute value as last r
                assert hasattr(policy_dict[aid], "value_functon")
                last_r[aid] = policy_dict[aid].value_function(episode, agent_key=aid)
    episodes = compute_value(episodes, policy_dict)
    episodes = compute_advantage(episodes, policy_dict, last_r=last_r, use_gae=True)
    return episode


def compute_value(
    episodes: List[Dict[str, Dict[AgentID, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
):
    for episode in episodes:
        for aid, policy in policy_dict.items():
            episode[aid][Episode.STATE_VALUE] = policy.value_function(**episode[aid])
    return episodes


# XXX(ming): require test
def copy_next_frame(
    episodes: List[Dict[AgentID, Dict[str, np.ndarray]]],
    policy_dict: Dict[AgentID, Policy],
):
    for episode in episodes:
        for aid, agent_episode in episode.items():
            assert Episode.CUR_OBS in agent_episode, (aid, episode)
            agent_episode[Episode.NEXT_OBS] = agent_episode[Episode.CUR_OBS].copy()

            if Episode.ACTION_MASK in agent_episode:
                agent_episode[Episode.NEXT_ACTION_MASK] = agent_episode[
                    Episode.ACTION_MASK
                ].copy()

            if Episode.CUR_STATE in agent_episode:
                agent_episode[Episode.NEXT_STATE] = agent_episode[
                    Episode.CUR_STATE
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
