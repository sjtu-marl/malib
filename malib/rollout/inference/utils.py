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

from gym import spaces

from malib.utils.typing import AgentID, DataFrame, EnvID
from malib.utils.episode import Episode
from malib.utils.preprocessor import Preprocessor
from malib.rollout.envs.vector_env import VectorEnv


def process_env_rets(
    env_rets: List[Tuple["states", "observations", "rewards", "dones", "infos"]],
    preprocessor: Dict[AgentID, Preprocessor],
    preset_meta_data: Dict[str, Any],
):
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
    env_rets_list_to_save = []
    env_dones = []

    # env_ret: state, obs, rew, done, info
    for ret in env_rets:
        # state, obs, reward, done, info
        agents = list(ret[1].keys())

        processed_obs = {
            agent: preprocessor[agent].transform(raw_obs)
            for agent, raw_obs in ret[1].items()
        }

        all_agents.update(agents)
        for agent, obs in processed_obs.items():
            agent_obs_list[agent].append(obs)

        env_rets_to_save = {Episode.CUR_OBS: processed_obs}

        if ret[0] is not None:
            for agent, _state in ret[0].items():
                agent_state_list[agent].append(_state)
            env_rets_to_save[Episode.CUR_STATE] = ret[0]

        original_obs_space = list(preprocessor.values())[0].original_space
        if (
            isinstance(original_obs_space, spaces.Dict)
            and "action_mask" in original_obs_space
        ):
            action_mask = {}
            for agent, env_obs in ret[1].items():
                agent_action_mask_list[agent].append(env_obs["action_mask"])
                action_mask[agent] = env_obs["action_mask"]
            env_rets_to_save[Episode.ACTION_MASK] = action_mask

        env_rets_to_save[Episode.PRE_REWARD] = ret[2]
        for agent, done in ret[3].items():
            if agent == "__all__":
                env_done = done
                continue
            agent_dones_list[agent].append(done)
        env_rets_to_save[Episode.PRE_DONE] = {
            k: v for k, v in ret[3].items() if k != "__all__"
        }

        env_dones.append(env_done)
        env_rets_list_to_save.append(env_rets_to_save)

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
                "env_num": len(env_rets_list_to_save),
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

    return env_dones, env_rets_list_to_save, dataframes


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

    rets = []  # env_id, str, agent, any
    for dataframes in raw_output.values():
        for dataframe in dataframes:
            agent = dataframe.identifier
            data = dataframe.data
            env_num = dataframe.meta_data["env_num"]

            if len(rets) == 0:
                rets = [defaultdict(dict) for _ in range(env_num)]

            assert isinstance(data, dict)

            for k, v in data.items():
                if k == Episode.RNN_STATE:
                    for i in range(env_num):
                        if v is None:
                            continue
                        rets[i][agent][k] = [_v[i] for _v in v]
                else:
                    for i, _v in enumerate(v):
                        rets[i][agent][k] = _v

    env_actions: Dict[AgentID, np.ndarray] = env.action_adapter(rets)

    return env_actions, rets
