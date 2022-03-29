import traceback
from collections import defaultdict
from types import LambdaType
import ray
import time
import gym
import os
import numpy as np
import gc
import copy

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
    Union,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.episode import Episode, EpisodeKey, NewEpisodeDict
from malib.utils.preprocessor import get_preprocessor
from malib.utils.timing import Timing
from malib.algorithm.common.policy import Policy
from malib.envs.vector_env import VectorEnv
from malib.envs.async_vector_env import AsyncVectorEnv, AsyncSubProcVecEnv
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
    for runtime_id, v in queue.items():
        rets[runtime_id] = v.get_nowait()
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
    replaced_holder = {}
    env_ids = list(env_rets.keys())
    preprocessor = server_runtime_config["preprocessor"]

    for env_id, ret in env_rets.items():
        processed[env_id] = {}
        replaced_holder[env_id] = {}

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
                if dk not in replaced_holder[env_id]:
                    replaced_holder[env_id][dk] = {}

                dataframes[agent][dk].append(_v)
                processed[env_id][k][agent] = _v
                replaced_holder[env_id][dk][agent] = _v

    # pack to data frame
    for env_aid in dataframes.keys():
        dataframes[env_aid] = DataFrame(
            identifier=env_aid,
            data={_k: np.stack(_v).squeeze() for _k, _v in dataframes[env_aid].items()},
            runtime_config={
                "behavior_mode": server_runtime_config["behavior_mode"],
                "environment_ids": env_ids,
            },
        )
        # check batch size:
        pred_batch_size = list(dataframes[env_aid].data.values())[0].shape[0]
        for _k, _v in dataframes[env_aid].data.items():
            assert _v.shape[0] == pred_batch_size, (_k, _v.shape, pred_batch_size)

    return replaced_holder, processed, dataframes


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
                if k == EpisodeKey.RNN_STATE:
                    # split to each environment
                    for i, env_id in enumerate(env_ids):
                        rets[env_id][k][agent] = [_v[i] for _v in v]
                else:
                    for env_id, _v in zip(env_ids, v):
                        # if k == EpisodeKey.ACTION:
                        #     print("------------ action dim:", _v.shape)
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
        training_agent_mapping: LambdaType = None,
    ):
        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()
        self.timer = Timing()
        self.training_agent_mapping = training_agent_mapping or (lambda agent: agent)

        agent_group = defaultdict(lambda: [])
        runtime_agent_ids = []
        for agent in env_desc["possible_agents"]:
            runtime_id = training_agent_mapping(agent)
            agent_group[runtime_id].append(agent)
            runtime_agent_ids.append(runtime_id)
        self.runtime_agent_ids = set(runtime_agent_ids)
        self.agent_group = dict(agent_group)

        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]
        env_cls = env_desc["creator"]
        env_config = env_desc["config"]

        self.preprocessor = {
            agent: get_preprocessor(obs_spaces[agent])(obs_spaces[agent])
            for agent in env_desc["possible_agents"]
        }

        if use_subproc_env:
            self.env = AsyncSubProcVecEnv(obs_spaces, act_spaces, env_cls, env_config)
        else:
            self.env = AsyncVectorEnv(obs_spaces, act_spaces, env_cls, env_config)

            # build connection with agent interfaces
        self.recv_queue = None
        self.send_queue = None

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

    def close(self):
        if self.recv_queue is not None:
            _ = [e.shutdown(force=True) for e in self.recv_queue.values()]
            _ = [e.shutdown(force=True) for e in self.send_queue.values()]
        self.env.close()

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces: Dict[AgentID, InferenceWorkerSet],
        desc: Dict[str, Any],
        buffer_desc: Union[BufferDescription, Dict[AgentID, BufferDescription]],
        reset: bool = False,
    ) -> Tuple[str, Dict[str, List]]:

        # desc required:
        #   flag,
        #   behavior_policies,
        #   policy_distribution (optional),
        #   parameter_desc_dict
        #   num_episodes,
        #   max_step (optional)
        #   postprocessor_types

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = desc["flag"]

        server_runtime_config = {
            "behavior_mode": None,
            "main_behavior_policies": desc["behavior_policies"],
            "policy_distribution": desc.get("policy_distribution", None),
            "sample_mode": "once",
            "parameter_desc_dict": desc["parameter_desc_dict"],
            "preprocessor": self.preprocessor,
        }

        client_runtime_config = {
            "max_step": desc["max_step"],
            "fragment_length": desc["max_step"] * desc["num_episodes"],
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

        if self.recv_queue is None or reset:
            self.recv_queue = {
                runtime_id: Queue(actor_options={"num_cpus": 0.1})
                for runtime_id in agent_interfaces
            }
            self.send_queue = {
                runtime_id: Queue(actor_options={"num_cpus": 0.1})
                for runtime_id in agent_interfaces
            }

        with self.timer.timeit("inference_server_connect"):
            _ = ray.get(
                [
                    server.connect.remote(
                        [self.recv_queue[runtime_id], self.send_queue[runtime_id]],
                        runtime_config=server_runtime_config,
                        runtime_id=self.process_id,
                    )
                    for runtime_id, server in agent_interfaces.items()
                ],
                timeout=3.0,
            )
            gc.collect()

        self.add_envs(desc["num_episodes"])

        try:
            with self.timer.timeit("environment_reset"):
                rets = self.env.reset(
                    limits=client_runtime_config["num_envs"],
                    fragment_length=client_runtime_config["fragment_length"],
                    max_step=client_runtime_config["max_step"],
                    custom_reset_config=client_runtime_config["custom_reset_config"],
                    # trainable_mapping=client_runtime_config["trainable_mapping"],
                )

            # TODO(ming): process env returns here
            _, rets, dataframes = process_env_rets(rets, server_runtime_config)
            episodes = NewEpisodeDict(lambda env_id: Episode(None, env_id=env_id))

            assert (
                len(self.env.active_envs) == client_runtime_config["num_envs"]
            ), self.env.active_envs.keys()

            start = time.time()
            while not self.env.is_terminated():
                with self.timer.time_avg("policy_step"):
                    grouped_data_frames = defaultdict(lambda: [])
                    for agent, dataframe in dataframes.items():
                        # map runtime to agent
                        runtime_id = self.training_agent_mapping(agent)
                        grouped_data_frames[runtime_id].append(dataframe)
                    for runtime_id, _send_queue in self.send_queue.items():
                        _send_queue.put_nowait(grouped_data_frames[runtime_id])
                    policy_outputs = recieve(self.recv_queue)
                    # print("env step:", self.env.batched_step_cnt, len(self.env.active_envs), client_runtime_config["fragment_length"], task_type)
                    env_actions, policy_outputs = process_policy_outputs(
                        policy_outputs, self.env
                    )
                with self.timer.time_avg("environment_step"):
                    next_rets = self.env.step(env_actions)
                    assert len(next_rets) > 0, env_actions
                    # merge RNN states here
                    rets_holder, next_rets, next_dataframes = process_env_rets(
                        next_rets, server_runtime_config
                    )
                    # collect all rets and save as buffer
                    # env_id, k, agent_ids
                    env_rets = merge_env_rets(rets, next_rets)
                assert len(env_rets) > 0
                episodes.record(policy_outputs, env_rets)
                # update next keys
                rets = rets_holder
                dataframes = next_dataframes
            end = time.time()

            rollout_info = self.env.collect_info()
            if task_type == "rollout":
                episodes = list(
                    episodes.to_numpy(
                        self.batch_mode, filter=list(desc["behavior_policies"].keys())
                    ).values()
                )
                # episodes = postprocessing(episodes, desc["postprocessor_types"])
                for runtime_id, _buffer_desc in buffer_desc.items():
                    _buffer_desc.batch_size = (
                        self.env.batched_step_cnt
                        if self.batch_mode == "time_step"
                        else len(episodes)
                    )

                    indices = None
                    while indices is None:
                        batch = ray.get(
                            self.dataset_server.get_producer_index.remote(_buffer_desc)
                        )
                        indices = batch.data
                    gc.collect()

                    _buffer_desc.data = []
                    for e in episodes:
                        for agent in self.agent_group[runtime_id]:
                            _buffer_desc.data.append({runtime_id: e[agent]})
                    _buffer_desc.indices = indices
                    self.dataset_server.save.remote(_buffer_desc)
        except Exception as e:
            traceback.print_exc()
            raise e

        ph = list(rollout_info.values())

        holder = {}
        for history, ds, k, vs in iter_many_dicts_recursively(*ph, history=[]):
            arr = [np.sum(_vs) for _vs in vs]
            prefix = "/".join(history)
            holder[prefix] = np.mean(arr)

        performance = self.timer.todict()
        performance["FPS"] = self.env.batched_step_cnt / (end - start)
        # print("performance:", performance)
        res = {
            "task_type": task_type,
            "total_fragment_length": self.env.batched_step_cnt,
            "eval_info": holder,
            "performance": performance,
        }
        return res
