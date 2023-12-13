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

import ray
import numpy as np

from malib.utils.typing import AgentID
from malib.rollout.episode import ConventionalEpisodeList

from malib.utils.timing import Timing
from malib.utils.data import merge_array_by_keys
from malib.remote.interface import RemoteInterface
from malib.rollout.config import RolloutConfig
from malib.rollout.inference.client import InferenceClient, PolicyReturnWithObs
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

    def merge_episodes(
        self, agent_groups: Dict[str, Tuple]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """A dict of merged episodes, which is grouped by agent groups.

        Args:
            agent_groups (Dict[str, Tuple]): A dict of agent groups.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: A dict of merged episodes, which is grouped by agent groups.
        """

        episodes: List[Dict[AgentID, Dict[str, np.ndarray]]] = self.episodes.to_numpy()

        # then merge this episodes by agent groups
        merged = {}
        for episode in episodes:
            for gid, agents in agent_groups.items():
                filtered = [episode[agent] for agent in agents]
                # then merge them by keys
                tmp = merge_array_by_keys(filtered)
                merged[gid] = tmp
        return merged


from malib.utils.timing import Timing
from malib.backend.dataset_server.utils import send_data


class BasicEnvRunner(RemoteInterface):
    def __repr__(self) -> str:
        return super().__repr__()

    def __init__(
        self,
        env_func: Type,
        max_env_num: int,
        use_subproc_env: bool = False,
        agent_groups: Dict[str, Tuple] = None,
        inference_entry_points: Dict[str, str] = None,
    ) -> None:
        super(RemoteInterface, self).__init__()

        self._use_subproc_env = use_subproc_env
        self._max_env_num = max_env_num
        self._env_func = env_func
        self._envs = []
        self._agent_groups = agent_groups
        self._inference_entry_points = inference_entry_points
        self._inference_clients = None

    @property
    def inference_clients(self) -> Dict[str, ray.ObjectRef]:
        return self._inference_clients

    @property
    def envs(self) -> Tuple[Environment]:
        return tuple(self._envs)

    @property
    def env_func(self) -> Type:
        return self._env_func

    @property
    def agent_groups(self) -> Dict[str, Tuple]:
        return self._agent_groups

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
        rollout_config: RolloutConfig,
        strategy_specs: Dict[AgentID, StrategySpec],
        inference_clients: Dict[AgentID, InferenceClient] = None,
        data_entrypoints: Dict[str, str] = None,
    ):
        """Single thread env simulation stepping.

        Args:
            rollout_config (RolloutConfig): Rollout configuration, which specifies how many data pieces will rollout.
            strategy_specs (Dict[AgentID, StrategySpec]): A dict of strategy specs, which rules the behavior policy of each agent.
            inference_clients (Dict[AgentID, InferenceClient]): A dict of remote inference client, mapping from env agents to inference clients. Note that there could be a shared client for multiple agents.
            data_entrypoints (Dict[str, str], optional): A mapping which defines the data collection trigger, if not None, then return episodes. Defaults to None.

        Raises:
            e: _description_

        Returns:
            _type_: _description_
        """

        if inference_clients is None:
            assert (
                self._inference_entry_points is not None
            ), "Inference client namespace should be specified if infer_clients is not given."
            assert (
                self._agent_groups is not None
            ), "Agent groups should be specified if infer_clients is not given."
            if self.inference_clients is None:
                res = {}
                for rid, _agents in self._agent_groups.items():
                    namespace, name = self._inference_entry_points[rid].split(":")
                    client = ray.get_actor(name=name, namespace=namespace)
                    res.update(dict.fromkeys(_agents, client))
                self._inference_clients = res
            inference_clients = self.inference_clients

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
        data = agent_manager.merge_episodes(agent_groups=self.agent_groups)
        data_entrypoints = data_entrypoints or {}
        for k, entrypoint in data_entrypoints.items():
            send_data(data[k], entrypoint=entrypoint)

        stats = {"total_timesteps": total_timestep, **timer.todict()}
        return stats
