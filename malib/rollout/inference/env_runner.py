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
from typing import Any, List, Dict, Tuple, Set, Type
from types import LambdaType
from collections import defaultdict

import os
import time
import traceback

import pickle
import ray

from ray.actor import ActorHandle

from malib.utils.typing import AgentID, DataFrame, BehaviorMode
from malib.utils.episode import ConventionalEpisodeList
from malib.utils.preprocessor import Preprocessor, get_preprocessor
from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterface
from malib.rollout.envs.vector_env import VectorEnv, SubprocVecEnv
from malib.common.rollout_config import RolloutConfig
from malib.rollout.inference.client import InferenceClient
from malib.rollout.inference.utils import process_env_rets, process_policy_outputs
from malib.rollout.envs.env import Environment
from malib.backend.dataset_server.utils import send_data


class AgentManager:
    def __init__(self, episode_num, inference_clients) -> None:
        self.inference_clients = inference_clients
        self.episodes = ConventionalEpisodeList(
            num=episode_num, agents=list(inference_clients.keys())
        )

    def collect_and_act(self, episode_idx, raw_obs, last_dones, last_rews, states):
        if not last_dones["__all__"]:
            action_and_obs = {
                k: ray.get(v.compute_action.remote(raw_obs[k], states[k]))
                for k, v in self.inference_clients.items()
            }
            actions = {}
            obs = {}
            for k, v in action_and_obs.items():
                actions[k] = v[0]
                obs[k] = v[1]
        else:
            actions = None
            obs = {
                k: ray.get(v.preprocess_obs.remote(raw_obs[k]))
                for k, v in self.inference_clients.items()
            }

        self.episodes.record(obs, last_dones, last_rews, states, episode_idx)

        return actions

    def merge_episodes(self):
        return self.episodes.to_numpy()


class BasicEnvRunner(RemoteInterface):
    def __repr__(self) -> str:
        return super().__repr__()

    def __init__(
        self, env_func: Type, max_env_num: int, use_subproc_env: bool = False
    ) -> None:
        super().__init__()

        self._use_subproc_env = use_subproc_env
        self._max_env_num = max_env_num
        self._env_func = env_func
        self._envs = []

    @property
    def envs(self) -> Tuple[Environment]:
        return tuple(self._envs)

    @property
    def env_func(self) -> Type:
        return self._env_func

    @property
    def num_active_envs(self) -> int:
        return len(self._envs)

    @property
    def use_subproc_env(self) -> bool:
        return self._use_subproc_env

    @property
    def max_env_num(self) -> int:
        return self._max_env_num

    def run(
        self,
        inference_clients: Dict[AgentID, InferenceClient],
        rollout_config: RolloutConfig,
        data_entrypoint_mapping: Dict[AgentID, str] = None,
    ):
        """Single thread env simulation stepping.

        Args:
            inference_clients (Dict[AgentID, InferenceClient]): A dict of remote inference client.
            rollout_config (RolloutConfig): Rollout configuration, which specifies how many data pieces will rollout.
            data_entrypoint_mapping (Dict[AgentID, str], optional): A mapping which defines the data collection trigger, if not None, then return episodes. Defaults to None.

        Raises:
            e: _description_

        Returns:
            _type_: _description_
        """

        new_env_num = max(0, rollout_config.n_envs_per_worker - self.num_active_envs)

        for _ in range(new_env_num):
            self._envs.append(self.env_func())

        # reset envs
        envs = self.envs[: rollout_config.n_envs_per_worker]
        vec_states, vec_obs, vec_dones, vec_rews = [], [], [], []

        for env in envs:
            states, obs = env.reset(max_step=rollout_config.timelimit)
            vec_states.append(states)
            vec_obs.append(obs)
            vec_dones.append(False)
            vec_rews.append(0.0)

        active_env_num = len(envs)
        agent_manager = AgentManager(active_env_num, inference_clients)

        while active_env_num:
            for env_idx, (env, states, obs, dones, rews) in enumerate(
                zip(envs, vec_states, vec_obs, vec_dones, vec_rews)
            ):
                if env.is_deactivated():
                    continue

                actions = agent_manager.collect_and_act(
                    env_idx,
                    raw_obs=obs,
                    last_dones=dones,
                    last_rews=rews,
                    states=states,
                )

                if actions is None:
                    # which means done already
                    active_env_num -= 1
                    env.set_done()
                else:
                    states, obs, rews, dones = env.step(actions)
                    # update frames
                    vec_states[env_idx] = states
                    vec_obs[env_idx] = obs
                    vec_dones[env_idx] = dones
                    vec_rews[env_idx] = rews

        # merge agent episodes
        data = agent_manager.merge_episodes()
        return data


class EnvRunner(RemoteInterface):
    def __repr__(self) -> str:
        return f"<EnvRunner max_env_num={self.max_env_num}, batch_mode={self.batch_mode}, use_subproc_env={self.use_subproc_env}>"

    def __init__(
        self,
        env_desc: Dict[str, Any],
        max_env_num: int,
        agent_groups: Dict[str, Set],
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: LambdaType = None,
        custom_config: Dict[str, Any] = {},
    ):
        """Construct an inference client, one for each agent.

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

        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()
        self.timer = Timing()
        self.training_agent_mapping = training_agent_mapping or (lambda agent: agent)
        self.max_env_num = max_env_num
        self.custom_configs = custom_config
        self.runtime_agent_ids = list(agent_groups.keys())
        self.agent_groups = agent_groups

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
        inference_clients: Dict[AgentID, InferenceClient],
        rollout_config: Dict[str, Any],
        data_entrypoint_mapping: Dict[AgentID, str] = None,
    ) -> Dict[str, Any]:
        """Executes environment runner to collect training data or run purely simulation/evaluation.

        Note:
            Only simulation/evaluation tasks return evaluation information.

        Args:
            inference_clients (Dict[AgentID, InferenceClient]): A dict of agent interface servers.
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

        eval_results, performance = _env_runner(
            self,
            inference_clients,
            self.preprocessor,
            rollout_config,
            server_runtime_config,
            data_entrypoint_mapping,
        )

        res = performance.copy()
        if task_type != "rollout":
            res["evaluation"] = eval_results
        return res


def _env_runner(
    client: InferenceClient,
    agents: Dict[str, InferenceClient],
    preprocessors: Dict[str, Preprocessor],
    rollout_config: Dict[str, Any],
    server_runtime_config: Dict[str, Any],
    data_entrypoint_mapping: Dict[AgentID, str],
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
    evaluate_on = rollout_config["behavior_mode"] == BehaviorMode.EXPLOITATION
    remote_actor = isinstance(list(agents.values())[0], ActorHandle)

    try:
        if data_entrypoint_mapping is not None:
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
            preprocessors=preprocessors,
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
                            agent.compute_action.remote(
                                grouped_data_frames[rid],
                                runtime_config=server_runtime_config,
                            )
                        )
                        for rid, agent in agents.items()
                    }
                else:
                    policy_outputs: Dict[str, List[DataFrame]] = {
                        rid: agent.compute_action(
                            grouped_data_frames[rid],
                            runtime_config=server_runtime_config,
                        )
                        for rid, agent in agents.items()
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

        if data_entrypoint_mapping is not None:
            # episode_id: agent_id: dict_data
            episodes = episodes.to_numpy()
            for entrypoint in data_entrypoint_mapping.values():
                send_data(pickle.dumps(episodes), entrypoint)
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
