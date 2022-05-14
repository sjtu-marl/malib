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

from typing import Type, Union, Any, List, Dict, Tuple
from types import LambdaType
from collections import defaultdict

import time
import os
import gc
import traceback

import ray
import numpy as np
import reverb

from ray.util.queue import Queue
from reverb.client import Writer as ReverbWriter

from malib import settings
from malib.remote.interface import RemoteInterFace
from malib.utils.logger import Log
from malib.utils.typing import (
    AgentID,
    BufferDescription,
    DataFrame,
    BehaviorMode,
    EnvID,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.episode import Episode, EpisodeKey, NewEpisodeDict
from malib.utils.preprocessor import get_preprocessor
from malib.utils.timing import Timing
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
    remain_env_ids = []
    preprocessor = server_runtime_config["preprocessor"]

    for env_id, ret in env_rets.items():
        processed[env_id] = {}
        replaced_holder[env_id] = {}

        dframe_jump = any(ret.get(EpisodeKey.DONE, {"default": False}).values())
        if not dframe_jump:
            remain_env_ids.append(env_id)

        for k, agent_v in ret.items():
            processed[env_id][k] = {}

            dk = k  # key for dataframes
            if k == EpisodeKey.DONE:
                agent_v.pop("__all__")
            for agent, _v in agent_v.items():
                if not dframe_jump and (agent not in dataframes):
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

                if not dframe_jump:
                    dataframes[agent][dk].append(_v)
                    replaced_holder[env_id][dk][agent] = _v
                processed[env_id][k][agent] = _v

    # pack to data frame
    for env_aid in dataframes.keys():
        dataframes[env_aid] = DataFrame(
            identifier=env_aid,
            data={_k: np.stack(_v).squeeze() for _k, _v in dataframes[env_aid].items()},
            runtime_config={
                "behavior_mode": server_runtime_config["behavior_mode"],
                "environment_ids": remain_env_ids,
            },
        )

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


def postprocessing(episodes, postprocessor_types, policies=None):
    postprocessor_types = ["default"]
    for handler in get_postprocessor(postprocessor_types):
        episodes = handler(episodes, policies)
    return episodes


class InferenceClient(RemoteInterFace):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: LambdaType = None,
    ):
        """Construct an inference client.

        Args:
            env_desc (Dict[str, Any]): Environment description
            dataset_server (_type_): A ray object reference.
            max_env_num (int): The maximum of created environment instance.
            use_subproc_env (bool, optional): Indicate subproc envrionment enabled or not. Defaults to False.
            batch_mode (str, optional): Batch mode, could be `time_step` or `episode` mode. Defaults to "time_step".
            postprocessor_types (Dict, optional): Post processor type list. Defaults to None.
            training_agent_mapping (LambdaType, optional): Agent mapping function. Defaults to None.
        """

        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()
        self.timer = Timing()
        self.training_agent_mapping = training_agent_mapping or (lambda agent: agent)
        self.max_env_num = max_env_num

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
            self.env = AsyncSubProcVecEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )
        else:
            self.env = AsyncVectorEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )

        self.recv_queue = None
        self.send_queue = None
        self.reverb_clients: Dict[str, Type[reverb.Client]] = {}

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these \
            new environment instances into it,otherwise do nothing.

        Args:
            maximum (int): Maximum limits.

        Returns:
            int: The number of nested environments.
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

    def run(
        self,
        agent_interfaces: Dict[AgentID, InferenceWorkerSet],
        desc: Dict[str, Any],
        dataserver_entrypoint: str = None,
        reset: bool = False,
    ) -> Union[List, Dict]:

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = desc["flag"]

        mapped_parameter_desc_dict = {}
        for rid in agent_interfaces.keys():
            agents = self.agent_group[rid]
            # random choice
            mapped_parameter_desc_dict[rid] = {
                aid: desc["parameter_desc_dict"][aid] for aid in agents
            }

        server_runtime_config = {
            "behavior_mode": None,
            # a mapping, from agent to pid
            "main_behavior_policies": desc["behavior_policies"],
            "sample_mode": "once",
            "preprocessor": self.preprocessor,
        }

        client_runtime_config = {
            "max_step": desc["max_step"],
            "fragment_length": desc["fragment_length"],
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
                runtime_id: Queue(actor_options={"num_cpus": 0})
                for runtime_id in agent_interfaces
            }
            self.send_queue = {
                runtime_id: Queue(actor_options={"num_cpus": 0})
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
                timeout=10.0,
            )
            gc.collect()

        if dataserver_entrypoint is not None:
            if dataserver_entrypoint not in self.reverb_clients:
                with self.timer.timeit("dataset_sever_connect"):
                    address = ray.get(
                        self.dataset_server.get_client_kwargs.remote(
                            dataserver_entrypoint
                        )
                    )["address"]
                    self.reverb_clients[dataserver_entrypoint] = reverb.Client(address)
            reverb_writer: ReverbWriter = self.reverb_clients[
                dataserver_entrypoint
            ].writer(max_sequence_length=desc["fragment_length"])

        else:
            reverb_writer: ReverbWriter = None

        try:
            with self.timer.timeit("environment_reset"):
                rets = self.env.reset(
                    fragment_length=client_runtime_config["fragment_length"],
                    max_step=client_runtime_config["max_step"],
                    custom_reset_config=client_runtime_config["custom_reset_config"],
                )

            _, rets, dataframes = process_env_rets(rets, server_runtime_config)
            episodes = NewEpisodeDict(lambda env_id: Episode(None, env_id=env_id))

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
                    env_rets = merge_env_rets(rets, next_rets)
                assert len(env_rets) > 0
                # episodes.record(policy_outputs, env_rets)
                if reverb_writer is not None:
                    reverb_writer.append()
                # update next keys
                rets = rets_holder
                dataframes = next_dataframes
            end = time.time()
            rollout_info = self.env.collect_info()
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
        res = {
            "task_type": task_type,
            "total_timesteps": self.env.batched_step_cnt,
            "performance": performance,
        }
        if task_type in ["evaluation", "simulation"]:
            res["evaluation"] = holder
        return res
