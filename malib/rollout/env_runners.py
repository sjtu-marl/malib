from typing import Callable, Dict, Any, List, Tuple
from argparse import Namespace
from collections import defaultdict

import time
import traceback

import numpy as np

from malib.utils.episode import Episode, NewEpisodeDict
from malib.utils.typing import AgentID, DataFrame, EnvID
from malib.envs.vector_env import VectorEnv
from malib.rollout.inference_client import InferenceClient, recieve


def process_env_rets(
    env_rets: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    server_runtime_config: Dict[str, Any],
) -> Dict[AgentID, DataFrame]:
    """Process environment returns, generally, for the observation transformation.

    Args:
        env_rets (Dict[EnvID, Dict[str, Dict[AgentID, Any]]]): A dict of environment returns.
        server_runtime_config (Dict[str, Any]): _description_

    Returns:
        Dict[AgentID, DataFrame]: _description_
    """

    processed = {}
    dataframes = {}
    replaced_holder = {}
    remain_env_ids = []
    preprocessor = server_runtime_config["preprocessor"]

    agent_obs_list = defaultdict(lambda: [])
    agent_action_mask_list = defaultdict(lambda: [])
    agent_dones_list = defaultdict(lambda: [])

    all_agents = set()
    for env_id, ret in env_rets.items():
        # process obs
        for agent, raw_obs in ret[0]:
            agent_obs_list[agent].append(preprocessor[agent].transform(raw_obs))
        agents = list(ret[0].keys())
        all_agents.update(agents)
        if len(ret) > 2:
            for agent, done in ret[2]:
                agent_dones_list[agent].append(done)
        else:
            for agent in agents:
                agent_dones_list[agent].append(False)
        for agent, action_mask in ret[-1]:
            agent_action_mask_list[agent].append(action_mask)

    for agent in all_agents:
        dataframes[agent] = DataFrame(
            identifier=agent,
            data={
                Episode.CUR_OBS: np.stack(agent_obs_list[agent]),
                Episode.ACTION_MASK: np.stack(agent_action_mask_list[agent]),
                Episode.DONE: np.stack(agent_dones_list[agent]),
            },
        )

    return dataframes


def process_policy_outputs(
    raw_output: Dict[str, List[DataFrame]], env: VectorEnv
) -> Tuple[None, Dict[EnvID, Dict[str, Dict[AgentID, Any]]]]:
    """Processing policy outputs for each agent.

    :param raw_output: A dict of raw policy output, mapping from agent to a data frame which is bound to a remote inference server.
    :type raw_output: Dict[AgentID, DataFrame]
    :return: A dict of dict, mapping from episode key to a cleaned agent dict
    :rtype: Dict[str, Dict[AgentID, Any]]
    """

    rets = defaultdict(lambda: defaultdict(lambda: {}))  # env_id, str, agent, any
    for dataframes in raw_output.values():
        # data should be a dict of agent value
        for dataframe in dataframes:
            agent = dataframe.identifier
            data = dataframe.data
            env_ids = dataframe.runtime_config["environment_ids"]

            assert isinstance(data, dict)

            for k, v in data.items():
                if k == Episode.RNN_STATE:
                    for i, env_id in enumerate(env_ids):
                        rets[env_id][k][agent] = [_v[i] for _v in v]
                else:
                    for env_id, _v in zip(env_ids, v):
                        rets[env_id][k][agent] = _v

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


def env_runner(
    client: InferenceClient,
    request: Namespace,
    server_runtime_config: Dict[str, Any],
    collection_backend: Callable = None,
):
    try:
        episode_dict = NewEpisodeDict(
            lambda: Episode(agents=client.env.possible_agents)
        )
        with client.timer.timeit("environment_reset"):
            env_rets = client.env.reset(
                fragment_length=request.fragment_length,
                max_step=request.max_step,
                custom_reset_config=request.custom_reset_config,
            )

        dataframes = process_env_rets(env_rets, server_runtime_config)
        episode_dict.record(env_rets)

        start = time.time()
        while not client.env.is_terminated():
            with client.timer.time_avg("policy_step"):
                grouped_data_frames = defaultdict(lambda: [])
                for agent, dataframe in dataframes.items():
                    # map runtime to agent
                    runtime_id = client.training_agent_mapping(agent)
                    grouped_data_frames[runtime_id].append(dataframe)
                for runtime_id, _send_queue in client.send_queue.items():
                    _send_queue.put_nowait(grouped_data_frames[runtime_id])
                policy_outputs = recieve(client.recv_queue)
                env_actions, policy_outputs = process_policy_outputs(
                    policy_outputs, client.env
                )

            with client.timer.time_avg("environment_step"):
                env_rets = client.env.step(env_actions)
                assert len(env_rets) > 0, env_actions
                # merge RNN states here
                dataframes = process_env_rets(env_rets, server_runtime_config)
                episode_dict.record(env_rets)

        if collection_backend is not None:
            # episode_id: agent_id: dict_data
            collection_backend(episodes=episode_dict.to_numpy())
        end = time.time()
        rollout_info = client.env.collect_info()
    except Exception as e:
        traceback.print_exc()
        raise e

    performance = client.timer.todict()
    performance["FPS"] = client.env.batched_step_cnt / (end - start)

    res = list(rollout_info.values())
    return res
    # for history, ds, k, vs in iter_many_dicts_recursively(*ph, history=[]):
    #     arr = [np.sum(_vs) for _vs in vs]
    #     prefix = "/".join(history)
    #     holder[prefix] = np.mean(arr)

    # res = {
    #     "task_type": task_type,
    #     "total_timesteps": client.env.batched_step_cnt,
    #     "performance": performance,
    # }
    # if task_type in ["evaluation", "simulation"]:
    #     res["evaluation"] = holder
