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
from typing import Type, Union, Any, List, Dict, Tuple, Callable
from types import LambdaType
from collections import defaultdict

import os
import gc
import time
import traceback

import ray
import reverb
import numpy as np

from ray.util.queue import Queue
from reverb.client import Writer as ReverbWriter

from malib.utils.typing import AgentID, DataFrame, EnvID, BehaviorMode
from malib.utils.episode import Episode, NewEpisodeDict
from malib.utils.preprocessor import Preprocessor, get_preprocessor
from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterFace
from malib.rollout.envs.vector_env import VectorEnv
from malib.rollout.envs.async_vector_env import AsyncVectorEnv, AsyncSubProcVecEnv
from malib.rollout.postprocessor import get_postprocessor
from malib.rollout.inference_server import InferenceWorkerSet


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

    dataframes = {}
    preprocessor = server_runtime_config["preprocessor"]

    agent_obs_list = defaultdict(lambda: [])
    agent_action_mask_list = defaultdict(lambda: [])
    agent_dones_list = defaultdict(lambda: [])

    all_agents = set()
    alive_env_ids = []
    for env_id, ret in env_rets.items():
        # obs, action_mask, reward, done, info
        # process obs
        agents = list(ret[0].keys())
        if len(ret) >= 2:
            for agent, action_mask in ret[1].items():
                agent_action_mask_list[agent].append(action_mask)
            # check done
            all_done = ret[3]["__all__"]
            if all_done:
                continue
            else:
                ret[3].pop("__all__")
                for agent, done in ret[3].items():
                    agent_dones_list[agent].append(done)
        else:
            for agent in agents:
                agent_dones_list[agent].append(False)
        for agent, raw_obs in ret[0].items():
            agent_obs_list[agent].append(preprocessor[agent].transform(raw_obs))
        all_agents.update(agents)
        alive_env_ids.append(env_id)

    server_runtime_config["environment_ids"] = alive_env_ids
    for agent in all_agents:
        dataframes[agent] = DataFrame(
            identifier=agent,
            data={
                Episode.CUR_OBS: np.stack(agent_obs_list[agent]),
                Episode.ACTION_MASK: np.stack(agent_action_mask_list[agent])
                if agent_action_mask_list.get(agent)
                else None,
                Episode.DONE: np.stack(agent_dones_list[agent]),
            },
            runtime_config=server_runtime_config,
        )

    return dataframes


def process_policy_outputs(
    raw_output: Dict[str, List[DataFrame]], env: VectorEnv
) -> Dict[EnvID, Dict[AgentID, Any]]:
    """Processing policy outputs for each agent.

    Args:
        raw_output (Dict[str, List[DataFrame]]): A dict of raw policy output, mapping from agent to a data frame which is bound to a remote inference server.
        env (VectorEnv): Environment instance.

    Returns:
        Dict[EnvID, Dict[AgentID, Any]]: Agent action by environments.
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
    ) -> Dict[str, Any]:
        """Run environment runner to collect training data or pure simulation.

        Args:
            agent_interfaces (Dict[AgentID, InferenceWorkerSet]): A dict of agent interface server.
            desc (Dict[str, Any]): Task description.
            dataserver_entrypoint (str, optional): Dataserver entrypoint, actually a reverb server name. Defaults to None.
            reset (bool, optional): Reset connection if existing connect detected. Defaults to False.

        Returns:
            Dict[str, Any]: Simulation results.
        """

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = desc["flag"]

        desc["custom_reset_config"] = desc.get("custom_reset_config", {})
        server_runtime_config = desc.copy()
        server_runtime_config.update(
            {
                "sample_mode": "once",
                # TODO(ming): move to policy
                "preprocessor": self.preprocessor,
            }
        )
        request = Namespace(**desc)

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
                    self.reverb_clients[dataserver_entrypoint] = reverb.Client(
                        f"localhost:{address}"
                    )
            reverb_writer: ReverbWriter = self.reverb_clients[
                dataserver_entrypoint
            ].writer(max_sequence_length=desc["fragment_length"])

        else:
            reverb_writer: ReverbWriter = None

        def collect_backend(episodes: Dict[EnvID, Dict[AgentID, Dict]]):
            print(f"collect activated with reverb writer: {reverb_writer}...")

        results = env_runner(
            self, request, server_runtime_config, collect_backend=collect_backend
        )
        return results


def env_runner(
    client: InferenceClient,
    request: Namespace,
    server_runtime_config: Dict[str, Any],
    collect_backend: Callable = None,
):
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
                env_actions, processed_policy_outputs = process_policy_outputs(
                    policy_outputs, client.env
                )
                episode_dict.record(processed_policy_outputs)

            with client.timer.time_avg("environment_step"):
                env_rets = client.env.step(env_actions)
                assert len(env_rets) > 0, env_actions
                # merge RNN states here
                dataframes = process_env_rets(env_rets, server_runtime_config)
                episode_dict.record(env_rets)

        if collect_backend is not None:
            # episode_id: agent_id: dict_data
            episodes = episode_dict.to_numpy()
            # print("episodes:", episode_dict.to_numpy())
            collect_backend(episodes=episodes)
        end = time.time()
        rollout_info = client.env.collect_info()
    except Exception as e:
        traceback.print_exc()
        raise e

    performance = client.timer.todict()
    performance["FPS"] = client.env.batched_step_cnt / (end - start)

    res = list(rollout_info.values())
    # TODO(ming): merge information?
    return res
