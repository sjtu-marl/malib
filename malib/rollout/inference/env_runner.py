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

from typing import Any, List, Dict, Tuple, Set, Type
from types import LambdaType
from collections import defaultdict

import os
import time
import traceback

import pickle
import ray
import numpy as np

from ray.actor import ActorHandle

from malib.utils.typing import AgentID, DataFrame, BehaviorMode
from malib.utils.episode import ConventionalEpisodeList

from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterface
from malib.rollout.envs.vector_env import VectorEnv, SubprocVecEnv
from malib.rollout.rollout_config import RolloutConfig
from malib.rollout.inference.client import InferenceClient, PolicyReturnWithObs
from malib.rollout.inference.utils import process_env_rets, process_policy_outputs
from malib.rollout.envs.env import Environment
from malib.common.strategy_spec import StrategySpec
from malib.backend.dataset_server.utils import send_data


class AgentManager:
    def __init__(
        self,
        episode_num: int,
        inference_clients: Dict[AgentID, ray.ObjectRef],
        strategy_specs: Dict[AgentID, StrategySpec],
    ) -> None:
        """Construct a unified API for multi-agent action caller.

        Args:
            episode_num (int): Defines how many episodes will be collected, used for initialization.
            inference_clients (Dict[AgentID, ray.ObjectRef]): A dict of remote inference clients.
        """

        self.inference_clients = inference_clients
        self.strategy_specs = strategy_specs
        self.episodes = ConventionalEpisodeList(
            num=episode_num, agents=list(inference_clients.keys())
        )
        self.use_active_policy = dict.fromkeys(self.inference_clients.keys(), False)
        self.checkpoints = {}

    def set_behavior_policy(self):
        """Specify behavior policy for each agent."""

        for agent_id, strategy_spec in self.strategy_specs.items():
            if len(strategy_spec) == 0:
                self.use_active_policy[agent_id] = True
            else:
                self.checkpoints[agent_id] = strategy_spec.sample()

    def collect_and_act(
        self,
        episode_idx: int,
        raw_obs: Dict[AgentID, Any],
        last_dones: Dict[AgentID, bool],
        last_rews: Dict[AgentID, float],
        states: Dict[AgentID, np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Collect give timestep, if last_dones['__all__'] is True, then return a None action.

        Args:
            episode_idx (int): Episode index, for identifying episode buffer.
            raw_obs (Dict[AgentID, Any]): A dict of raw agent observations.
            last_dones (Dict[AgentID, bool]): A dict of agent dones, accompanying with __all__ to identify environment done.
            last_rews (Dict[AgentID, float]): A dict of rewards of last timestep.
            states (Dict[AgentID, np.ndarray], optional): A dict of states. Defaults to None.

        Returns:
            Dict[str, Any]: A dict of actions.
        """

        policy_return_with_obs: Dict[AgentID, PolicyReturnWithObs] = {
            k: ray.get(
                v.compute_action.remote(
                    raw_obs=raw_obs[k],
                    state=states[k] if states is not None else None,
                    last_reward=last_rews[k],
                    last_done=last_dones[k],
                    active_policy=self.use_active_policy[k],
                    checkpoint=self.checkpoints.get(k),
                )
            )
            for k, v in self.inference_clients.items()
        }
        actions = {}
        obs = {}
        for k, v in policy_return_with_obs.items():
            actions[k] = v.action
            obs[k] = v.obs

        self.episodes.record(obs, actions, last_dones, last_rews, states, episode_idx)

        return actions

    def merge_episodes(self):
        return self.episodes.to_numpy()


from malib.utils.timing import Timing


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
        strategy_specs: Dict[AgentID, StrategySpec],
        data_entrypoint_mapping: Dict[AgentID, str] = None,
    ):
        """Single thread env simulation stepping.

        Args:
            inference_clients (Dict[AgentID, InferenceClient]): A dict of remote inference client.
            rollout_config (RolloutConfig): Rollout configuration, which specifies how many data pieces will rollout.
            strategy_specs (Dict[AgentID, StrategySpec]): A dict of strategy specs, which rules the behavior policy of each agent.
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
            vec_dones.append(
                {"__all__": False, **dict.fromkeys(env.possible_agents, False)}
            )
            vec_rews.append(dict.fromkeys(env.possible_agents, 0.0))

        active_env_num = len(envs)
        agent_manager = AgentManager(active_env_num, inference_clients, strategy_specs)

        timer = Timing()
        total_timestep = 0

        while active_env_num:
            for env_idx, (env, states, obs, dones, rews) in enumerate(
                zip(envs, vec_states, vec_obs, vec_dones, vec_rews)
            ):
                if env.is_deactivated:
                    continue
                total_timestep += 1

                with timer.time_avg("avg_env_step"):
                    with timer.time_avg("avg_inference_client_step"):
                        actions = agent_manager.collect_and_act(
                            env_idx,
                            raw_obs=obs,
                            last_dones=dones,
                            last_rews=rews,
                            states=states,
                        )

                    if dones["__all__"]:
                        # which means done already
                        active_env_num -= 1
                        env.deactivate()
                    else:
                        states, obs, rews, dones, info = env.step(actions)
                        # update frames
                        vec_states[env_idx] = states
                        vec_obs[env_idx] = obs
                        vec_dones[env_idx] = dones
                        vec_rews[env_idx] = rews

        # merge agent episodes
        # FIXME(ming): send data to remote dataset
        data = agent_manager.merge_episodes()
        stats = {"total_timesteps": total_timestep, **timer.todict()}
        return stats


from malib.utils.episode import NewEpisodeList
from malib.utils.preprocessor import Preprocessor


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
