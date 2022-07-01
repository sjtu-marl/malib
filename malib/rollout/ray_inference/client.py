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

from argparse import Namespace
from typing import Any, List, Dict, Tuple
from types import LambdaType
from collections import defaultdict

import os
import time
import traceback

import ray
import numpy as np

from ray.util.queue import Queue
from ray.actor import ActorHandle

from malib.utils.logging import Logger

from malib.utils.typing import AgentID, DataFrame, EnvID, BehaviorMode
from malib.utils.episode import Episode, NewEpisodeDict
from malib.utils.preprocessor import Preprocessor, get_preprocessor
from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterface
from malib.rollout.envs.vector_env import VectorEnv
from malib.rollout.envs.async_vector_env import AsyncVectorEnv, AsyncSubProcVecEnv
from malib.rollout.postprocessor import get_postprocessor
from malib.rollout.ray_inference.server import RayInferenceWorkerSet


def process_env_rets(
    env_rets: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    preprocessor: Dict[AgentID, Preprocessor],
    preset_meta_data: Dict[str, Any],
) -> Dict[AgentID, DataFrame]:
    """Process environment returns, generally, for the observation transformation.

    Args:
        env_rets (Dict[EnvID, Dict[str, Dict[AgentID, Any]]]): A dict of environment returns.
        preprocessor (Dict[AgentID, Preprocessor]): A dict of preprocessor for raw environment observations, mapping from agent ids to preprocessors.
        preset_meta_data (Dict[str, Any]): Preset meta data.

    Returns:
        Dict[AgentID, DataFrame]: A dict of dataframes, mapping from agent ids to dataframes.
    """

    dataframes = {}
    agent_obs_list = defaultdict(lambda: [])
    agent_action_mask_list = defaultdict(lambda: [])
    agent_dones_list = defaultdict(lambda: [])

    all_agents = set()
    alive_env_ids = []
    processed_env_rets = {}

    for env_id, ret in env_rets.items():
        # obs, action_mask, reward, done, info
        # process obs
        agents = list(ret[0].keys())
        processed_obs = {
            agent: preprocessor[agent].transform(raw_obs)
            for agent, raw_obs in ret[0].items()
        }
        # obs, action_mask, reward, done, info,
        processed_env_rets[env_id] = (processed_obs,) + ret[1:]

        # check done
        if len(ret) > 2:
            all_done = ret[3]["__all__"]
            if all_done:
                continue
            else:
                ret[3].pop("__all__")
                for agent, done in ret[3].items():
                    agent_dones_list[agent].append(done)
        else:
            for agent in agents:
                if isinstance(ret[0][agent], Tuple):
                    agent_dones_list[agent].append([False] * len(ret[0][agent]))
                else:
                    agent_dones_list[agent].append(False)

        # do not move this inference before check done
        if len(ret) >= 2:
            for agent, action_mask in ret[1].items():
                agent_action_mask_list[agent].append(action_mask)

        for agent, obs in processed_obs.items():
            agent_obs_list[agent].append(obs)
        all_agents.update(agents)
        alive_env_ids.append(env_id)

    for agent in all_agents:
        stacked_obs = np.stack(agent_obs_list[agent])
        stacked_action_mask = (
            np.stack(agent_action_mask_list[agent])
            if agent_action_mask_list.get(agent)
            else None
        )
        stacked_done = np.stack(agent_dones_list[agent])

        dataframes[agent] = DataFrame(
            identifier=agent,
            data={
                Episode.CUR_OBS: stacked_obs,
                Episode.ACTION_MASK: stacked_action_mask,
                Episode.DONE: stacked_done,
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
                },
            },
        )

    return processed_env_rets, dataframes


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


def recieve(queue: Dict[str, Queue], block: bool = False) -> Dict[AgentID, DataFrame]:
    """Recieves message from remote servers. If block, then wait until not empty.

    Args:
        queue (Dict[str, Queue]): A dict of queues, mapping from runtime ids to queues.
        block (bool, optional): Sync mode or not. Defaults to False.

    Returns:
        Dict[AgentID, DataFrame]: A dict of frames, mapping from agent ids to dataframes.
    """

    rets = {}
    for runtime_id, v in queue.items():
        if block:
            rets[runtime_id] = v.get()
        else:
            if not v.empty():
                rets[runtime_id] = v.get_nowait()
    return rets


def postprocessing(episodes, postprocessor_types, policies=None):
    postprocessor_types = ["default"]
    for handler in get_postprocessor(postprocessor_types):
        episodes = handler(episodes, policies)
    return episodes


class RayInferenceClient(RemoteInterface):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server: ray.ObjectRef,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: LambdaType = None,
        custom_config: Dict[str, Any] = {},
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
            use_internal_server (bool, optional): Enabler internal server or not. Defaults to False.
            parameter_server (ray.ObjectRef, optional): If use internal server, the parameter server must be initialized.
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

        self.preprocessor: Dict[str, Preprocessor] = {
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

        use_internal_server: bool = (
            custom_config.get("inference_server", "local") == "local"
        )

        if use_internal_server:
            runtime_obs_spaces = {}
            runtime_act_spaces = {}

            for rid, agents in self.agent_group.items():
                runtime_obs_spaces[rid] = obs_spaces[agents[0]]
                runtime_act_spaces[rid] = act_spaces[agents[0]]

            self.agent_interfaces = {
                runtime_id: RayInferenceWorkerSet(
                    agent_id=runtime_id,
                    observation_space=runtime_obs_spaces[runtime_id],
                    action_space=runtime_act_spaces[runtime_id],
                    parameter_server=custom_config["parameter_server"],
                    governed_agents=self.agent_group[runtime_id],
                )
                for runtime_id in runtime_agent_ids
            }
        else:
            self.agent_interfaces = None

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
        """Disconnects with inference servers and turns off environment."""

        if self.recv_queue is not None:
            _ = [e.shutdown(force=True) for e in self.recv_queue.values()]
            _ = [e.shutdown(force=True) for e in self.send_queue.values()]
        self.env.close()

    def run(
        self,
        agent_interfaces: Dict[AgentID, RayInferenceWorkerSet],
        desc: Dict[str, Any],
        dataserver_entrypoint: str = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Executes environment runner to collect training data or run purely simulation/evaluation.

        Note:
            Only simulation/evaluation tasks return evaluation information.

        Args:
            agent_interfaces (Dict[AgentID, InferenceWorkerSet]): A dict of agent interface server.
            desc (Dict[str, Any]): Task description.
            dataserver_entrypoint (str, optional): Dataserver entrypoint, to identify online dataset servers for different experiments. Defaults to None.
            reset (bool, optional): Reset connection if existing connect detected. Defaults to False.

        Returns:
            Dict[str, Any]: A dict of simulation results.
        """

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = desc["flag"]

        desc["custom_reset_config"] = desc.get("custom_reset_config", {})
        server_runtime_config = desc.copy()
        server_runtime_config.update(
            {
                "sample_mode": "once",
                "preprocessor": self.preprocessor,
                "evaluate": task_type != "rollout",
            }
        )
        request = Namespace(**desc)

        if task_type == "rollout":
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLORATION
        elif task_type in ["evaluation", "simulation"]:
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLOITATION

        eval_results, performance = env_runner(
            self,
            self.agent_interfaces or agent_interfaces,
            request,
            server_runtime_config,
            writer_info_dict=desc.get("writer_info_dict", None),
        )

        res = performance.copy()
        if task_type != "rollout":
            res["evaluation"] = eval_results
        return res


def env_runner(
    client: RayInferenceClient,
    servers: Dict[str, RayInferenceWorkerSet],
    request: Namespace,
    server_runtime_config: Dict[str, Any],
    writer_info_dict: Dict[str, Tuple[str, Queue]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """The main logic of environment stepping, also for data collections.

    Args:
        client (InferenceClient): The inference client.
        request (Namespace): A namespace instance which describes the request.
        server_runtime_config (Dict[str, Any]): A dict which gives the runtime configuration of inference server.
        writer_info_dict (Dict[str, Tuple[str, Queue]], optional): A dict maps from runtime ids to a tuple of writer info. Defaults to None.

    Raises:
        e: General exceptions.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, float]]: A tuple of evaluation results and performance results.
    """

    # check whether remote server or not
    remote_actor = isinstance(list(servers.values())[0], ActorHandle)

    try:
        episode_dict = NewEpisodeDict(
            lambda: Episode(
                agents=client.env.possible_agents,
                processors=server_runtime_config["preprocessor"],
            )
        )

        with client.timer.timeit("environment_reset"):
            env_rets = client.env.reset(
                fragment_length=request.fragment_length,
                max_step=request.max_step,
                custom_reset_config=request.custom_reset_config,
            )

        processed_env_ret, dataframes = process_env_rets(
            env_rets,
            preprocessor=server_runtime_config["preprocessor"],
            preset_meta_data={"evaluate": server_runtime_config["evaluate"]},
        )
        # record environment return at reset has been called:
        # (obs, action_mask)
        episode_dict.record_env_rets(processed_env_ret)

        Logger.debug("env runner started...")
        start = time.time()
        # async policy interaction
        cnt = 0
        while not client.env.is_terminated():
            grouped_data_frames: Dict[str, List[DataFrame]] = defaultdict(lambda: [])
            for agent, dataframe in dataframes.items():
                # map runtime to agent
                runtime_id = client.training_agent_mapping(agent)
                grouped_data_frames[runtime_id].append(dataframe)

            with client.timer.time_avg("policy_step"):
                if remote_actor:
                    policy_outputs: Dict[str, List[DataFrame]] = {
                        rid: ray.get(
                            server.compute_action.remote(
                                grouped_data_frames[rid],
                                runtime_config=server_runtime_config,
                            )
                        )
                        for rid, server in servers.items()
                    }
                else:
                    policy_outputs: Dict[str, List[DataFrame]] = {
                        rid: server.compute_action(
                            grouped_data_frames[rid],
                            runtime_config=server_runtime_config,
                        )
                        for rid, server in servers.items()
                    }

            with client.timer.time_avg("process_policy_output"):
                env_actions, processed_policy_outputs = process_policy_outputs(
                    policy_outputs, client.env
                )

                episode_dict.record_policy_step(processed_policy_outputs)

            with client.timer.time_avg("environment_step"):
                env_rets = client.env.step(env_actions)
                if len(env_rets) < 1:
                    dataframes = {}
                    continue
                # merge RNN states here
                processed_env_ret, dataframes = process_env_rets(
                    env_rets,
                    preprocessor=server_runtime_config["preprocessor"],
                    preset_meta_data={"evaluate": server_runtime_config["evaluate"]},
                )
                episode_dict.record_env_rets(processed_env_ret)

            cnt += 1
            # FIXME(ming): only time step mode support
            # if writer_info_dict is not None and cnt % 100 == 0:
            #     episodes = episode_dict.to_numpy()
            #     for rid, writer_info in writer_info_dict.items():
            #         # get agents from agent group
            #         agents = client.agent_group[rid]
            #         batches = []
            #         for episode in episodes.values():
            #             agent_buffer = [episode[aid] for aid in agents]
            #             batches.append(agent_buffer)
            #         writer_info[-1].put_nowait_batch(batches)
            #     # rewrite
            #     episode_dict.clear()
            #     episode_dict.record_env_rets(
            #         processed_env_ret, ignore_keys={"rew", "infos", "done"}
            #     )
            #     print(
            #         "send data, current rollout fps: {:.3f}".format(
            #             client.env.batched_step_cnt / (time.time() - start)
            #         )
            #     )

        if writer_info_dict is not None:
            # episode_id: agent_id: dict_data
            episodes = episode_dict.to_numpy()
            for rid, writer_info in writer_info_dict.items():
                # get agents from agent group
                agents = client.agent_group[rid]
                batches = []
                for episode in episodes.values():
                    agent_buffer = [episode[aid] for aid in agents]
                    batches.append(agent_buffer)
                writer_info[-1].put_nowait_batch(batches)
        end = time.time()
        rollout_info = client.env.collect_info()
    except Exception as e:
        traceback.print_exc()
        raise e

    performance = client.timer.todict()
    performance["FPS"] = client.env.batched_step_cnt / (end - start)
    eval_results = list(rollout_info.values())
    performance["total_timesteps"] = client.env.batched_step_cnt

    return eval_results, performance
