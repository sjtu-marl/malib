from collections import defaultdict
import ray
import time
import gym
import os
import numpy as np

from ray.util.queue import Queue

from malib import settings
from malib.utils.logger import Logger, Log
from malib.utils.typing import (
    AgentID,
    Any,
    BufferDescription,
    DataFrame,
    List,
    PolicyID,
    Dict,
    Status,
    BehaviorMode,
    Tuple,
    EnvID,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.episode import Episode, EpisodeKey, NewEpisodeDict
from malib.utils.preprocessor import get_preprocessor
from malib.algorithm.common.policy import Policy
from malib.envs.vector_env import SubprocVecEnv, VectorEnv
from malib.envs.async_vector_env import AsyncVectorEnv
from malib.rollout.postprocessor import get_postprocessor
from malib.rollout.inference_server import InferenceWorkerSet


def wait_recv(recv_queue: Dict[str, Queue]):
    while True:
        ready = True
        for recv in recv_queue.values():
            if recv.empty():
                ready = False
        if ready:
            break
        # else:
        #     time.sleep(1)


def recieve(queue: Dict[str, Queue]) -> Dict[AgentID, DataFrame]:
    """Recieving messages from remote server.

    :param queue: A dict of queue.
    :type queue: Dict[str, Queue]
    """

    wait_recv(queue)

    rets = {}
    for k, v in queue.items():
        rets[k] = v.get_nowait()
    return rets


def process_env_rets(
    env_rets: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    server_runtime_config: Dict[str, Any],
) -> Dict[AgentID, DataFrame]:
    """Process environment returns, generally, for the observation transformation.

    :param env_rets: A dict of environment returns.
    :type env_rets: Dict[str, Any]
    :return: A dict of environmen returns, which is washed.
    :rtype: Dict[str, Any]
    """

    processed = {}
    dataframes = {}
    env_ids = list(env_rets.keys())
    preprocessor = server_runtime_config["preprocessor"]

    for env_id, ret in env_rets.items():
        processed[env_id] = {}
        for k, agent_v in ret.items():
            processed[env_id][k] = {}
            dk = k  # key for dataframes
            if k == EpisodeKey.DONE:
                agent_v.pop("__all__")
            for agent, _v in agent_v.items():
                if agent not in dataframes:
                    dataframes[agent] = defaultdict(lambda: [])
                if k in [EpisodeKey.CUR_OBS, EpisodeKey.NEXT_OBS]:
                    dk = EpisodeKey.CUR_OBS
                    _v = preprocessor[agent].transform(_v)
                elif k in [EpisodeKey.CUR_STATE, EpisodeKey.NEXT_STATE]:
                    dk = EpisodeKey.CUR_STATE
                elif k in [EpisodeKey.ACTION_MASK, EpisodeKey.NEXT_ACTION_MASK]:
                    dk = EpisodeKey.ACTION_MASK

                dataframes[agent][dk].append(_v)
                processed[env_id][k][agent] = _v

    # pack to data frame
    for k in dataframes.keys():
        # print("data frameskesf", k, dataframes[k].keys())
        dataframes[k] = DataFrame(
            header=None,
            data=dataframes[k],
            runtime_config={
                "behavior_mode": server_runtime_config["behavior_mode"],
                "environment_ids": env_ids,
            },
        )

    return processed, dataframes


def process_policy_outputs(
    raw_output: Dict[AgentID, DataFrame], env: VectorEnv
) -> Tuple[None, Dict[EnvID, Dict[str, Dict[AgentID, Any]]]]:
    """Processing policy outputs for each agent.

    :param raw_output: A dict of raw policy output, mapping from agent to a data frame which is bound to a remote inference server.
    :type raw_output: Dict[AgentID, DataFrame]
    :return: A dict of dict, mapping from episode key to a cleaned agent dict
    :rtype: Dict[str, Dict[AgentID, Any]]
    """

    rets = defaultdict(lambda: defaultdict(lambda: {}))  # env_id, str, agent, any
    for agent, dataframe in raw_output.items():
        # data should be a dict of agent value
        data = dataframe.data
        env_ids = dataframe.runtime_config["environment_ids"]

        assert isinstance(data, dict)

        for k, v in data.items():
            if k == EpisodeKey.RNN_STATE:
                # split to each environment
                for i, env_id in enumerate(env_ids):
                    rets[env_id][k][agent] = [_v[i] for _v in v]
            else:
                for env_id, _v in zip(env_ids, v):
                    rets[env_id][k][agent] = _v

    # process action with action adapter
    env_actions: Dict[EnvID, Dict[AgentID, Any]] = env.action_adapter(rets)

    return env_actions, rets


def postprocessing(episodes, postprocessor_types, policies=None):
    postprocessor_types = ["default"]
    for handler in get_postprocessor(postprocessor_types):
        episodes = handler(episodes, policies)
    return episodes


@ray.remote
class InferenceClient:
    def __init__(
        self,
        env_desc,
        dataset_server,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
    ):
        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()

        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]
        env_cls = env_desc["creator"]
        env_config = env_desc["config"]

        self.preprocessor = {
            agent: get_preprocessor(obs_spaces[agent])(obs_spaces[agent])
            for agent in env_desc["possible_agents"]
        }

        if use_subproc_env:
            self.env = SubprocVecEnv(
                obs_spaces, act_spaces, env_cls, env_config, max_num_envs=2
            )
        else:
            self.env = AsyncVectorEnv(obs_spaces, act_spaces, env_cls, env_config)

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these new environment instances into it,
        otherwise do nothing.

        :returns: The number of nested environments.
        """

        if not isinstance(self.env, VectorEnv):
            return 1

        existing_env_num = getattr(self.env, "num_envs", 1)

        if existing_env_num >= maximum:
            return self.env.num_envs

        self.env.add_envs(num=maximum - existing_env_num)

        return self.env.num_envs

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces: Dict[AgentID, InferenceWorkerSet],
        fragment_length: int,
        desc: Dict[str, Any],
        buffer_desc: BufferDescription = None,
    ) -> Tuple[str, Dict[str, List]]:

        task_type = desc["flag"]
        obs_spaces = self.env
        # compute behavior_policies here
        server_runtime_config = {
            "behavior_mode": None,
            "main_behavior_policies": desc["behavior_policies"],
            "policy_distribution": desc.get("policy_distribution", None),
            "sample_mode": "once",
            "parameter_desc_dict": desc["paramter_desc_dict"],
            "preprocessor": self.preprocessor,
        }

        client_runtime_config = {
            "max_step": desc.get("max_step", None),
            "fragment_length": fragment_length,
            "num_envs": desc["num_episodes"],
            "custom_reset_config": None,
            "trainable_mapping": desc["behavior_policies"]
            if task_type == "rollout"
            else None,
        }

        if task_type == "rollout":
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLORATION
        elif task_type in ["evaluation", "simulation"]:
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLOITATION

        # build connection with agent interfaces
        recv_queue = {agent: Queue() for agent in agent_interfaces}
        send_queue = {agent: Queue() for agent in agent_interfaces}

        _ = ray.get(
            [
                server.connect.remote(
                    [recv_queue[aid], send_queue[aid]],
                    runtime_config=server_runtime_config,
                    runtime_id=self.process_id,
                )
                for aid, server in agent_interfaces.items()
            ],
            timeout=3.0,
        )

        self.add_envs(desc["num_episodes"])

        rets = self.env.reset(
            limits=client_runtime_config["num_envs"],
            fragment_length=client_runtime_config["fragment_length"],
            max_step=client_runtime_config["max_step"],
            custom_reset_config=client_runtime_config["custom_reset_config"],
            trainable_mapping=client_runtime_config["trainable_mapping"],
        )

        # TODO(ming): process env returns here
        rets, dataframes = process_env_rets(rets, server_runtime_config)
        episodes = NewEpisodeDict(lambda env_id: Episode(None, env_id=env_id))

        start = time.time()
        while not self.env.is_terminated():
            # send query to servers
            request_start = time.time()
            # print("ready to send dataframes: {}".format(list(dataframes.keys())))
            for agent, dataframe in dataframes.items():
                send_queue[agent].put_nowait(dataframe)

            policy_outputs = recieve(recv_queue)
            request_end = time.time()
            # Logger.info(
            #     "receive policy outputs: {}".format(list(policy_outputs.keys()))
            # )
            # assigne outputs to linked env
            process_policy_start = time.time()
            env_actions, policy_outputs = process_policy_outputs(
                policy_outputs, self.env
            )
            process_policy_end = time.time()
            env_step_start = time.time()
            next_rets = self.env.step(env_actions)
            env_step_end = time.time()
            process_env_start = time.time()
            next_rets, dataframes = process_env_rets(next_rets, server_runtime_config)
            process_env_end = time.time()
            # collect all rets and save as buffer
            env_rets = {**rets, **next_rets}
            episodes.record(policy_outputs, env_rets)
            rets = next_rets
            # print(
            #     "cnt / fps: {} {} total_time: {}, request: {}, process_policL {}, env_step: {}, process_env: {}".format(
            #     self.env._step_cnt["team_0"],
            #     self.env._step_cnt["team_0"] / (time.time() - start),
            #     time.time() - request_start,
            #     request_end - request_start, process_policy_end - process_policy_start, env_step_end - env_step_start, process_env_end - process_env_start
            #     )
            # )
            # if self.env._step_cnt["team_0"] % 100 == 0:
            #     print("fps:", self.env._step_cnt["team_0"] / (time.time() - start))

        _ = [e.shutdown(force=True) for e in recv_queue.values()]
        _ = [e.shutdown(force=True) for e in send_queue.values()]

        rollout_info = self.env.collect_info()
        # if task_type == "rollout":
        #     episodes = list(
        #         episodes.to_numpy(
        #             self.batch_mode, filter=list(desc["behavior_policies"].keys())
        #         ).values()
        #     )
        #     episodes = postprocessing(episodes, desc["postprocessor_types"])
        #     buffer_desc.batch_size = (
        #         self.env.batched_step_cnt
        #         if self.batch_mode == "time_step"
        #         else len(episodes)
        #     )

        #     indices = None
        #     while indices is None:
        #         batch = ray.get(
        #             self.dataset_server.get_producer_index.remote(buffer_desc)
        #         )
        #         indices = batch.data

        #     buffer_desc.data = episodes
        #     buffer_desc.indices = indices
        #     self.dataset_server.save.remote(buffer_desc)

        ph = list(rollout_info.values())

        holder = {}
        for history, ds, k, vs in iter_many_dicts_recursively(*ph, history=[]):
            arr = [np.sum(_vs) for _vs in vs]
            prefix = "/".join(history)
            holder[prefix] = arr

        res = {"total_fragment_length": self.env.batched_step_cnt, "eval_info": holder}
        # print("totaoff", res)
        return res
