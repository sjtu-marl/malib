from collections import defaultdict

import numpy as np

from malib.utils.typing import AgentID, EnvID, Dict, Any, PolicyID, List


class EpisodeKey:
    """Unlimited buffer"""

    CUR_OBS = "observation"
    NEXT_OBS = "next_observation"
    ACTION = "action"
    ACTION_MASK = "action_mask"
    REWARD = "reward"
    DONE = "done"
    # XXX(ziyu): Change to 'logits' for numerical issues.
    ACTION_DIST = "action_logits"
    # XXX(ming): seems useless
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


class Episode:
    def __init__(self, behavior_policies: Dict[AgentID, PolicyID], env_id: str):
        self.policy_mapping = behavior_policies
        self.env_id = env_id

        self.agent_entry = defaultdict(lambda: {aid: [] for aid in self.policy_mapping})

    def __getitem__(self, __k: str) -> Dict[AgentID, List]:
        return self.agent_entry[__k]

    def __setitem__(self, __k: str, v: Dict[AgentID, List]) -> None:
        self.agent_entry[__k] = v

    def to_numpy(self) -> Dict[str, Dict[AgentID, np.ndarray]]:
        # switch agent key and episode key
        res = defaultdict(lambda: {})
        for ek, agent_v in self.agent_entry.items():
            if ek == EpisodeKey.RNN_STATE:
                continue
            for agent_id, v in agent_v.items():
                res[agent_id][ek] = np.asarray(v, dtype=np.float32)
        return res


class NewEpisodeDict(defaultdict):
    def __missing__(self, env_id):
        if self.default_factory is None:
            raise KeyError(env_id)
        else:
            ret = self[env_id] = self.default_factory(env_id)
            return ret

    def record(
        self, policy_outputs, env_outputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]]
    ):
        for env_id, policy_output in policy_outputs.items():
            for k, v in env_outputs[env_id].items():
                if k == "infos":
                    continue
                agent_slot = self[env_id][k]
                for aid, _v in v.items():
                    agent_slot[aid].append(_v)
            for k, v in policy_output.items():
                agent_slot = self[env_id][k]
                assert aid in agent_slot, agent_slot
                for aid, _v in v.items():
                    agent_slot[aid].append(_v)

    def to_numpy(self) -> Dict[EnvID, Dict[AgentID, Dict[str, np.ndarray]]]:
        return {k: v.to_numpy() for k, v in self.items()}
