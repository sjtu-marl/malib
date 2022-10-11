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

from typing import Any, List, Dict, Tuple
from collections import defaultdict

import numpy as np

from malib.utils.typing import AgentID, DataFrame, EnvID
from malib.utils.episode import Episode
from malib.utils.preprocessor import Preprocessor
from malib.rollout.envs.vector_env import VectorEnv


def process_env_rets(
    env_rets: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    preprocessor: Dict[AgentID, Preprocessor],
    preset_meta_data: Dict[str, Any],
) -> Tuple[Dict[EnvID, Tuple], Dict[AgentID, DataFrame]]:
    """Process environment returns, generally, for the observation transformation.

    Args:
        env_rets (Dict[EnvID, Dict[str, Dict[AgentID, Any]]]): A dict of environment returns.
        preprocessor (Dict[AgentID, Preprocessor]): A dict of preprocessor for raw environment observations, mapping from agent ids to preprocessors.
        preset_meta_data (Dict[str, Any]): Preset meta data.

    Returns:
        Tuple[Dict[EnvID, Tuple], Dict[AgentID, DataFrame]]: A tuple of saving env returns and a dict of dataframes, mapping from agent ids to dataframes.
    """

    # legal keys including: obs, state, reward, info
    # action mask is a feature in observation

    dataframes = {}
    agent_obs_list = defaultdict(lambda: [])
    agent_action_mask_list = defaultdict(lambda: [])
    agent_dones_list = defaultdict(lambda: [])
    agent_state_list = defaultdict(lambda: [])

    all_agents = set()
    alive_env_ids = []
    env_rets_to_save = {}

    for env_id, ret in env_rets.items():
        # state, obs, reward, done, info
        agents = list(ret[0].keys())
        processed_obs = {
            agent: preprocessor[agent].transform(raw_obs)
            for agent, raw_obs in ret[0].items()
        }
        if ret[0] is not None:
            for agent, _state in ret[0].items():
                agent_state_list[agent].append(_state)
        if "action_mask" in list(preprocessor.values())[0].original_space:
            # TODO(ming): handling action_mask here
            action_mask = {}
            for agent, env_obs in ret[1].items():
                agent_action_mask_list[agent].append(env_obs["action_mask"])
                action_mask[agent] = env_obs["action_mask"]
            env_rets_to_save[env_id] = (ret[0], processed_obs, action_mask) + ret[2:]
        else:
            env_rets_to_save[env_id] = (
                ret[0],
                processed_obs,
            ) + ret[2:]

        # check done
        if len(ret) > 2:
            all_done = ret[3]["__all__"]
            if all_done:
                continue  # environment is terminated, has no need to proced dones.
            else:
                ret[3].pop("__all__")
                for agent, done in ret[3].items():
                    agent_dones_list[agent].append(done)
        else:
            for agent in agents:
                # XXX(ming): tuple for group?
                if isinstance(ret[0][agent], Tuple):
                    agent_dones_list[agent].append([False] * len(ret[0][agent]))
                else:
                    agent_dones_list[agent].append(False)

        for agent, obs in processed_obs.items():
            agent_obs_list[agent].append(obs)

        # collect agents and alive env ids here
        all_agents.update(agents)
        alive_env_ids.append(env_id)

    for agent in all_agents:
        stacked_obs = np.stack(agent_obs_list[agent])
        stacked_action_mask = (
            np.stack(agent_action_mask_list[agent])
            if agent_action_mask_list.get(agent)
            else None
        )
        stacked_state = (
            np.stack(agent_state_list[agent]) if agent_state_list.get(agent) else None
        )
        stacked_done = np.stack(agent_dones_list[agent])

        # making dataframe as policy inputs
        dataframes[agent] = DataFrame(
            identifier=agent,
            data={
                Episode.CUR_OBS: stacked_obs,
                Episode.ACTION_MASK: stacked_action_mask,
                Episode.DONE: stacked_done,
                Episode.CUR_STATE: stacked_state,
            },
            meta_data={
                "environment_ids": alive_env_ids,
                "evaluate": preset_meta_data["evaluate"],
                "data_shapes": {
                    Episode.CUR_OBS: stacked_obs.shape[1:],
                    Episode.ACTION_MASK: stacked_action_mask.shape[1:]
                    if stacked_action_mask is not None
                    else None,
                    Episode.DONE: stacked_done.shape[1:],
                    Episode.CUR_STATE: stacked_state.shape[1:]
                    if stacked_state is not None
                    else None,
                },
            },
        )

    return env_rets_to_save, dataframes


def process_policy_outputs(
    raw_output: Dict[str, List[DataFrame]], env: VectorEnv
) -> Tuple[Dict[EnvID, Dict[AgentID, Any]], Dict[EnvID, Dict[AgentID, Dict[str, Any]]]]:
    """Processing policy outputs for each agent.

    Args:
        raw_output (Dict[str, List[DataFrame]]): A dict of raw policy output, mapping from agent to a data frame which is bound to a remote inference server.
        env (VectorEnv): Environment instance.

    Returns:
        Tuple[Dict[EnvID, Dict[AgentID, Any]], Dict[EnvID, Dict[AgentID, Dict[str, Any]]]]: A tuple of 1. Agent action by environments, 2.
    """

    rets = defaultdict(lambda: defaultdict(lambda: {}))  # env_id, str, agent, any
    for dataframes in raw_output.values():
        for dataframe in dataframes:
            agent = dataframe.identifier
            data = dataframe.data
            env_ids = dataframe.meta_data["environment_ids"]
            assert isinstance(data, dict)
            for k, v in data.items():
                if k == Episode.RNN_STATE:
                    for i, env_id in enumerate(env_ids):
                        if v is None:
                            continue
                        rets[env_id][agent][k] = [_v[i] for _v in v]
                else:
                    for env_id, _v in zip(env_ids, v):
                        rets[env_id][agent][k] = _v

    # process action with action adapter
    env_actions: Dict[EnvID, Dict[AgentID, Any]] = env.action_adapter(rets)

    return env_actions, rets


def merge_env_rets(rets, next_rets):
    r: Dict[EnvID, Dict] = {}
    for e in [rets, next_rets]:
        for env_id, ret in e.items():
            if env_id not in r:
                r[env_id] = ret
            else:
                r[env_id].update(ret)
    return r
