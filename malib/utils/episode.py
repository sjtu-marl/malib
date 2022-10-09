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

from typing import Dict, Any, List, Sequence
from collections import defaultdict

import traceback
import numpy as np

from malib.utils.typing import AgentID, EnvID


class Episode:
    """Agent episode tracking"""

    CUR_OBS = "obs"
    NEXT_OBS = "obs_next"
    ACTION = "act"
    ACTION_MASK = "act_mask"
    NEXT_ACTION_MASK = "act_mask_next"
    REWARD = "rew"
    DONE = "done"
    ACTION_LOGITS = "act_logits"
    ACTION_DIST = "act_dist"
    INFO = "infos"

    # optional
    STATE_VALUE = "state_value_estimation"
    STATE_ACTION_VALUE = "state_action_value_estimation"
    CUR_STATE = "state"  # current global state
    NEXT_STATE = "state_next"  # next global state
    LAST_REWARD = "last_reward"

    # post process
    ACC_REWARD = "accumulate_reward"
    ADVANTAGE = "advantage"
    STATE_VALUE_TARGET = "state_value_target"

    # model states
    RNN_STATE = "rnn_state"

    def __init__(self, agents: List[AgentID], processors):
        self.processors = processors
        self.agents = agents
        self.agent_entry = {agent: defaultdict(lambda: []) for agent in self.agents}

    def __getitem__(self, __k: str) -> Dict[AgentID, List]:
        return self.agent_entry[__k]

    def __setitem__(self, __k: str, v: Dict[AgentID, List]) -> None:
        self.agent_entry[__k] = v

    def record_policy_step(self, policy_rets: Dict[AgentID, Dict[str, Any]]):
        """Save policy outputs.

        Args:
            policy_rets (Dict[AgentID, Dict[str, Any]]): A dict of policy outputs, each for an agent.
        """

        for agent, agent_item in policy_rets.items():
            for k, v in agent_item.items():
                self.agent_entry[agent][k].append(v)

    def record_env_rets(self, env_rets: Sequence[Dict[AgentID, Any]], ignore_keys={}):
        """Save a transiton. The given transition is a sub sequence of (obs, action_mask, reward, done, info). Users specify ignore keys to filter keys.

        Args:
            env_rets (Sequence[Dict[AgentID, Any]]): A transition.
            ignore_keys (dict, optional): . Defaults to {}.
        """

        for i, key in enumerate(
            [
                Episode.CUR_OBS,
                Episode.ACTION_MASK,
                Episode.REWARD,
                Episode.DONE,
                Episode.INFO,
            ]
        ):
            if len(env_rets) < i + 1:
                break
            if key in ignore_keys:
                continue
            for agent, _v in env_rets[i].items():
                if agent == "__all__":
                    continue
                self.agent_entry[agent][key].append(_v)

    def to_numpy(self) -> Dict[AgentID, Dict[str, np.ndarray]]:
        """Convert episode to numpy array-like data."""

        res = {}
        for agent, agent_trajectory in self.agent_entry.items():
            tmp = {}
            try:
                for k, v in agent_trajectory.items():
                    if k == Episode.CUR_OBS:
                        # move to next obs
                        tmp[Episode.NEXT_OBS] = v[1:]
                        tmp[Episode.CUR_OBS] = v[:-1]
                    elif k == Episode.CUR_STATE:
                        # move to next state
                        tmp[Episode.NEXT_STATE] = v[1:]
                        tmp[Episode.CUR_STATE] = v[:-1]
                    elif k == Episode.INFO:
                        continue
                    elif k == Episode.ACTION_MASK:
                        tmp[Episode.ACTION_MASK] = v[:-1]
                        tmp[Episode.NEXT_ACTION_MASK] = v[1:]
                    else:
                        tmp[k] = v
            except Exception as e:
                print(traceback.format_exc())
                raise e
            res[agent] = tmp
        # agent trajectory length check
        for agent, trajectory in res.items():
            assert "rew" in trajectory, trajectory.keys()
            expected_length = len(trajectory[Episode.CUR_OBS])
            for k, v in trajectory.items():
                assert len(v) == expected_length, (len(v), k, expected_length)
        return dict(res)


class NewEpisodeDict(defaultdict):
    """Episode dict, for trajectory tracking for a bunch of environments."""

    def __missing__(self, env_id):
        if self.default_factory is None:
            raise KeyError(env_id)
        else:
            ret = self[env_id] = self.default_factory()
            return ret

    def record_env_rets(
        self, env_outputs: Dict[EnvID, Sequence[Dict[AgentID, Any]]], ignore_keys={}
    ):
        """Save returns of environments.

        Args:
            env_outputs (Dict[EnvID, Sequence[Dict[AgentID, Any]]]): A dict of environment returns.
            ignore_keys (dict, optional): Oh, I forget the functionality. Defaults to {}.
        """

        for env_id, env_output in env_outputs.items():
            self[env_id].record_env_rets(env_output, ignore_keys)

    def record_policy_step(
        self, env_policy_outputs: Dict[EnvID, Dict[AgentID, Dict[str, Any]]]
    ):
        """Save policy outputs.

        Args:
            env_policy_outputs (Dict[EnvID, Dict[AgentID, Dict[str, Any]]]): A dict of policy outputs, each item is a dict related to an environment.
        """

        for env_id, policy_outputs in env_policy_outputs.items():
            self[env_id].record_policy_step(policy_outputs)

    def to_numpy(self) -> Dict[EnvID, Dict[AgentID, Dict[str, np.ndarray]]]:
        """Lossy data transformer, which converts a dict of episode to a dict of numpy array like. (some episode may be empty)"""

        res = {}
        for k, v in self.items():
            tmp: Dict[AgentID, Dict[str, np.ndarray]] = v.to_numpy()
            if len(tmp) == 0:
                continue
            res[k] = tmp
        return res
