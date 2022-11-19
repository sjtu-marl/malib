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

from ray.util.queue import Queue
from ray.actor import ActorHandle

from malib.utils.logging import Logger

from malib.utils.typing import AgentID, DataFrame, BehaviorMode
from malib.utils.episode import Episode, NewEpisodeDict, NewEpisodeList
from malib.utils.preprocessor import Preprocessor, get_preprocessor
from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterface
from malib.rollout.envs.vector_env import VectorEnv, SubprocVecEnv
from malib.rollout.inference.ray.server import RayInferenceWorkerSet
from malib.rollout.inference.utils import process_env_rets, process_policy_outputs


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
            custom_config (Dict[str, Any], optional): Custom configuration. Defaults to an empty dict.
        """

        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()
        self.timer = Timing()
        self.training_agent_mapping = training_agent_mapping or (lambda agent: agent)
        self.max_env_num = max_env_num
        self.custom_configs = custom_config

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
            self.env = SubprocVecEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )
        else:
            self.env = VectorEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )

    def close(self):
        """Disconnects with inference servers and turns off environment."""

        if self.recv_queue is not None:
            _ = [e.shutdown(force=True) for e in self.recv_queue.values()]
            _ = [e.shutdown(force=True) for e in self.send_queue.values()]
        self.env.close()

    def run(
        self,
        agent_interfaces: Dict[AgentID, RayInferenceWorkerSet],
        rollout_config: Dict[str, Any],
        dataset_writer_info_dict: Dict[str, Tuple[str, Queue]] = None,
    ) -> Dict[str, Any]:
        """Executes environment runner to collect training data or run purely simulation/evaluation.

        Note:
            Only simulation/evaluation tasks return evaluation information.

        Args:
            agent_interfaces (Dict[AgentID, InferenceWorkerSet]): A dict of agent interface servers.
            rollout_config (Dict[str, Any]): Rollout configuration.
            dataset_writer_info_dict (Dict[str, Tuple[str, Queue]], optional): Dataset writer info dict. Defaults to None.

        Returns:
            Dict[str, Any]: A dict of simulation results.
        """

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = rollout_config["flag"]

        server_runtime_config = {
            "preprocessor": self.preprocessor,
            "strategy_specs": rollout_config["strategy_specs"],
        }

        if task_type == "rollout":
            assert (
                dataset_writer_info_dict is not None
            ), "rollout task has no available dataset writer"
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLORATION
        elif task_type in ["evaluation", "simulation"]:
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLOITATION

        eval_results, performance = env_runner(
            self,
            agent_interfaces,
            rollout_config,
            server_runtime_config,
            dwriter_info_dict=dataset_writer_info_dict,
        )

        res = performance.copy()
        if task_type != "rollout":
            res["evaluation"] = eval_results
        return res


def env_runner(
    client: RayInferenceClient,
    servers: Dict[str, RayInferenceWorkerSet],
    rollout_config: Dict[str, Any],
    server_runtime_config: Dict[str, Any],
    dwriter_info_dict: Dict[str, Tuple[str, Queue]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """The main logic of environment stepping, also for data collections.

    Args:
        client (InferenceClient): The inference client.
        rollout_config (Dict[str, Any]): Rollout configuration.
        server_runtime_config (Dict[str, Any]): A dict which gives the runtime configuration of inference server. Keys including

            - `preprocessor`: observation preprocessor.
            - `behavior_mode`: a value of `BehaviorMode`.
            - `strategy_spec`: a dict of strategy specs, mapping from runtime agent id to strategy spces.

        dwriter_info_dict (Dict[str, Tuple[str, Queue]], optional): A dict maps from runtime ids to a tuple of dataset writer info. Defaults to None.

    Raises:
        e: General exceptions.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, float]]: A tuple of evaluation results and performance results.
    """

    # check whether remote server or not
    evaluate_on = server_runtime_config["behavior_mode"] == BehaviorMode.EXPLOITATION
    remote_actor = isinstance(list(servers.values())[0], ActorHandle)

    try:
        if dwriter_info_dict is not None:
            episodes = NewEpisodeList(
                num=client.env.num_envs, agents=client.env.possible_agents
            )
        else:
            episodes = None

        with client.timer.timeit("environment_reset"):
            env_rets = client.env.reset(
                fragment_length=rollout_config["fragment_length"],
                max_step=rollout_config["max_step"],
            )

        env_dones, processed_env_ret, dataframes = process_env_rets(
            env_rets=env_rets,
            preprocessor=server_runtime_config["preprocessor"],
            preset_meta_data={"evaluate": evaluate_on},
        )
        # env ret is key first, not agent first: state, obs
        if episodes is not None:
            episodes.record(
                processed_env_ret, agent_first=False, is_episode_done=env_dones
            )

        start = time.time()
        cnt = 0

        while not client.env.is_terminated():
            # group dataframes by runtime ids.
            grouped_data_frames: Dict[str, List[DataFrame]] = defaultdict(lambda: [])
            for agent, dataframe in dataframes.items():
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
                # TODO(ming): do not use async stepping
                env_actions, processed_policy_outputs = process_policy_outputs(
                    policy_outputs, client.env
                )

                if episodes is not None:
                    episodes.record(
                        processed_policy_outputs,
                        agent_first=True,
                        is_episode_done=env_dones,
                    )

            with client.timer.time_avg("environment_step"):
                env_rets = client.env.step(env_actions)
                env_dones, processed_env_ret, dataframes = process_env_rets(
                    env_rets=env_rets,
                    preprocessor=server_runtime_config["preprocessor"],
                    preset_meta_data={"evaluate": evaluate_on},
                )
                # state, obs, rew, done
                if episodes is not None:
                    episodes.record(
                        processed_env_ret, agent_first=False, is_episode_done=env_dones
                    )

            cnt += 1

        if dwriter_info_dict is not None:
            # episode_id: agent_id: dict_data
            episodes = episodes.to_numpy()
            for rid, writer_info in dwriter_info_dict.items():
                # get agents from agent group
                agents = client.agent_group[rid]
                batches = []
                # FIXME(ming): multi-agent is wrong!
                for episode in episodes:
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
    eval_results = rollout_info
    performance["total_timesteps"] = client.env.batched_step_cnt

    return eval_results, performance
