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
    """Multi-agent episode tracking"""

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

    def __init__(self, agents: List[AgentID], processors=None):
        # self.processors = processors
        self.agents = agents
        self.agent_entry = {agent: defaultdict(lambda: []) for agent in self.agents}

    def __getitem__(self, __k: AgentID) -> Dict[str, List]:
        """Return an agent dict.

        Args:
            __k (AgentID): Registered agent id.

        Returns:
            Dict[str, List]: A dict of transitions.
        """

        return self.agent_entry[__k]

    def __setitem__(self, __k: AgentID, v: Dict[str, List]) -> None:
        """Set an agent episode.

        Args:
            __k (AgentID): Agent ids
            v (Dict[str, List]): Transition dict.
        """

        self.agent_entry[__k] = v

    def record(
        self, data: Dict[str, Dict[str, Any]], agent_first: bool, ignore_keys={}
    ):
        """Save a transiton. The given transition is a sub sequence of (obs, action_mask, reward, done, info). Users specify ignore keys to filter keys.

        Args:
            data (Dict[str, Dict[AgentID, Any]]): A transition.
            ignore_keys (dict, optional): . Defaults to {}.
        """

        if agent_first:
            for agent, kvs in data.items():
                for k, v in kvs.items():
                    self.agent_entry[agent][k].append(v)
        else:
            for k, agent_trans in data.items():
                for agent, _v in agent_trans.items():
                    self.agent_entry[agent][k].append(_v)

    def to_numpy(self) -> Dict[AgentID, Dict[str, np.ndarray]]:
        """Convert episode to numpy array-like data."""

        res = {}
        for agent, agent_trajectory in self.agent_entry.items():
            tmp = {}
            try:
                for k, v in agent_trajectory.items():
                    if k == Episode.CUR_OBS:
                        # move to next obs
                        tmp[Episode.NEXT_OBS] = np.stack(v[1:])
                        tmp[Episode.CUR_OBS] = np.stack(v[:-1])
                    elif k == Episode.CUR_STATE:
                        # move to next state
                        tmp[Episode.NEXT_STATE] = np.stack(v[1:])
                        tmp[Episode.CUR_STATE] = np.stack(v[:-1])
                    elif k == Episode.ACTION_MASK:
                        tmp[Episode.ACTION_MASK] = np.stack(v[:-1])
                        tmp[Episode.NEXT_ACTION_MASK] = np.stack(v[1:])
                    else:
                        tmp[k] = np.stack(v)
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

    def record(
        self,
        data: Dict[EnvID, Dict[str, Dict[str, Any]]],
        agent_first: bool,
        ignore_keys={},
    ):
        for env_id, _data in data.items():
            self[env_id].record(_data, agent_first, ignore_keys)

    def to_numpy(self) -> Dict[EnvID, Dict[AgentID, Dict[str, np.ndarray]]]:
        """Lossy data transformer, which converts a dict of episode to a dict of numpy array like. (some episode may be empty)"""

        res = {}
        for k, v in self.items():
            tmp: Dict[AgentID, Dict[str, np.ndarray]] = v.to_numpy()
            if len(tmp) == 0:
                continue
            res[k] = tmp
        return res
