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

import numpy as np

from malib.utils.typing import AgentID, EnvID


class Episode:
    CUR_OBS = "observation"
    NEXT_OBS = "next_observation"
    ACTION = "action"
    ACTION_MASK = "action_mask"
    NEXT_ACTION_MASK = "next_action_mask"
    REWARD = "reward"
    DONE = "done"
    ACTION_LOGITS = "action_logits"
    ACTION_DIST = "action_dist"
    INFO = "infos"

    # optional
    STATE_VALUE = "state_value_estimation"
    STATE_ACTION_VALUE = "state_action_value_estimation"
    CUR_STATE = "state"  # current global state
    NEXT_STATE = "next_state"  # next global state
    LAST_REWARD = "last_reward"

    # post process
    ACC_REWARD = "accumulate_reward"
    ADVANTAGE = "advantage"
    STATE_VALUE_TARGET = "state_value_target"

    # model states
    RNN_STATE = "rnn_state"

    def __init__(self, agents, processors):
        self.processors = processors
        # self.agent_entry = defaultdict(lambda: {aid: [] for aid in self.policy_mapping})
        self.agents = agents
        self.agent_entry = {agent: defaultdict(lambda: []) for agent in self.agents}

    def __getitem__(self, __k: str) -> Dict[AgentID, List]:
        return self.agent_entry[__k]

    def __setitem__(self, __k: str, v: Dict[AgentID, List]) -> None:
        self.agent_entry[__k] = v

    def record(self, env_rets: Sequence[Dict[AgentID, Any]]):
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
            for agent, _v in env_rets[i].items():
                if agent == "__all__":
                    continue
                self.agent_entry[agent][key].append(_v)

    def to_numpy(self) -> Dict[AgentID, Dict[str, np.ndarray]]:
        """Convert episode to numpy array-like data."""

        res = {}
        for agent, agent_trajectory in self.agent_entry.items():
            tmp = {}
            for k, v in agent_trajectory.items():
                if k == Episode.CUR_OBS:
                    # move to next obs
                    tmp[Episode.NEXT_OBS] = np.asarray(v[1:])
                    tmp[Episode.CUR_OBS] = np.asarray(v[:-1])
                elif k == Episode.CUR_STATE:
                    # move to next state
                    tmp[Episode.NEXT_STATE] = np.asarray(v[1:])
                    tmp[Episode.CUR_STATE] = np.asarray(v[:-1])
                elif k == Episode.INFO:
                    continue
                else:
                    tmp[k] = np.asarray(v)
            res[agent] = tmp
        return dict(res)


class NewEpisodeDict(defaultdict):
    def __missing__(self, env_id):
        if self.default_factory is None:
            raise KeyError(env_id)
        else:
            ret = self[env_id] = self.default_factory()
            return ret

    def record(self, env_outputs: Dict[EnvID, Sequence[Dict[AgentID, Any]]]):
        for env_id, env_output in env_outputs.items():
            self[env_id].record(env_output)

    def to_numpy(self) -> Dict[EnvID, Dict[AgentID, Dict[str, np.ndarray]]]:
        """Lossy data transformer, which converts a dict of episode to a dict of numpy array like. (some episode may be empty)"""

        res = {}
        for k, v in self.items():
            tmp: Dict[AgentID, Dict[str, np.ndarray]] = v.to_numpy()
            if len(tmp) == 0:
                continue
            res[k] = tmp
        return res
