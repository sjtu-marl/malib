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
        return {
            k: dict(
                map(lambda kv: (kv[0], np.asarray(kv[1], dtype=np.float16)), v.items())
            )
            for k, v in self.agent_entry.items()
        }


class NewEpisodeDict(defaultdict):
    def __missing__(self, env_id):
        if self.default_factory is None:
            raise KeyError(env_id)
        else:
            ret = self[env_id] = self.default_factory(env_id)
            return ret
    # XXX(ziyu): We can remove NEXT_OBS to improve efficiency.
    def record(
        self, policy_outputs, env_outputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]]
    ):
        for env_id, env_output in env_outputs.items():
            for k, v in policy_outputs[env_id].items():
                # # FIXME(ziyu): @ming here I found that rnn_state
                # # which is a list of two rnn_state(actor/critic) 
                # # But some procedure will has dropped one of them.
                # if k == EpisodeKey.RNN_STATE:
                #     import pdb; pdb.set_trace()
                agent_slot = self[env_id][k]
                for aid, _v in v.items():
                    assert not isinstance(_v, Dict), (k, v)
                    agent_slot[aid].append(_v)
            for k, v in env_output.items():
                if k == "infos":
                    continue
                agent_slot = self[env_id][k]
                assert aid in agent_slot, agent_slot
                for aid, _v in v.items():
                    agent_slot[aid].append(_v)

    def to_numpy(self) -> Dict[EnvID, Dict[AgentID, Dict[str, np.ndarray]]]:
        return {k: v.to_numpy() for k, v in self.items()}
