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

from collections import ChainMap, defaultdict
from typing import Tuple, Dict, Any, List, Type, Callable, Sequence

import uuid

import gym
import ray
import numpy as np

from ray.actor import ActorHandle

from malib.utils.logging import Logger
from malib.utils.typing import (
    AgentID,
    EnvID,
    PolicyID,
)
from malib.rollout.envs.env import Environment
from malib.utils.episode import Episode


EnvironmentType = Type[Environment]


class VectorEnv:
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        preset_num_envs: int = 0,
    ):
        """Create a vector environment instance.

        Args:
            observation_spaces (Dict[AgentID, gym.Space]): A dict of agent observation spaces.
            action_spaces (Dict[AgentID, gym.Space]): A dict of agent action spaces.
            creator (type): Environment creator.
            configs (Dict[str, Any]): Environment configuration.
            preset_num_envs (int, optional): The number of started envrionments. Defaults to 0.
        """

        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.possible_agents = list(observation_spaces.keys())
        self.cached_episode_infos = {}
        self.fragment_length = None
        self.step_cnt = 0

        self._creator = creator
        self._configs = configs.copy()
        self._envs: List[Environment] = []
        self._action_adapter = creator.action_adapter

        self.add_envs(num=preset_num_envs)

    @property
    def batched_step_cnt(self) -> int:
        return self.step_cnt

    @property
    def num_envs(self) -> int:
        """The total number of environments"""

        return len(self._envs)

    @property
    def envs(self) -> List[Environment]:
        """Return a limited list of enviroments"""

        return self._envs

    @property
    def env_creator(self) -> EnvironmentType:
        """Return env creator.

        Returns:
            EnviornmentType: Class type
        """

        return self._creator

    @property
    def env_configs(self) -> Dict[str, Any]:
        """Return a copy of environment configuration for the construction.

        Returns:
            Dict[str, Any]: A dict of configuration.
        """

        return self._configs.copy()

    @classmethod
    def from_envs(cls, envs: List[Environment], config: Dict[str, Any]) -> "VectorEnv":
        """Construct a vectorenv from a give environment list.

        Args:
            envs (List[Environment]): A list of environment instances.
            config (Dict[str, Any]): A dict of environment configuration.

        Returns:
            VectorEnv: A VectorEnv instance.
        """

        observation_spaces = envs[0].observation_spaces
        action_spaces = envs[0].action_spaces

        vec_env = cls(observation_spaces, action_spaces, type(envs[0]), config, 0)
        vec_env.add_envs(envs=envs)

        return vec_env

    def add_envs(self, envs: List = None, num: int = 0):
        """Add exisiting `envs` or `num` new environments to this vectorization environment. If `envs` is not empty or None, the `num` will be ignored.

        Examples:
            >>> vector_env = VectorEnv(observation_spaces, action_spaces, creator, configs, preset_num_envs=0)
            >>>
            >>> # add existing environments to vectorenv
            >>> envs = [creator(**configs) for _ in range(5)]
            >>> vector_env.add_envs(envs=envs)
            >>> print(vector_env.num_envs)
            >>> 5
            >>>
            >>> # add new environments with specified number
            >>> vector_env.add_envs(num=4)
            >>> print(vector_env.num_envs)
            >>> 9

        Args:
            envs (List, optional): A list of environment. Defaults to None.
            num (int, optional): Number of enviornments need to be created. Defaults to 0.
        """

        if envs and len(envs) > 0:
            for env in envs:
                self.envs.append(env)
            Logger.debug(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            for _ in range(num):
                self.envs.append(self.env_creator(**self.env_configs))
            Logger.debug(f"created {num} new environments.")

    def reset(
        self,
        fragment_length: int,
        max_step: int,
    ) -> List[Tuple["states", "observations"]]:
        """Reset a bunch of environments.

        Args:
            fragment_length (int): Total timesteps before the VectorEnv is terminated.
            max_step (int): Maximum of episode length for each environment.

        Returns:
            Dict[EnvID, Sequence[Dict[AgentID, Any]]]: A dict of environment returns.
        """

        self.step_cnt = 0
        self.fragment_length = fragment_length
        self.max_step = max_step
        self.cached_episode_infos = []

        ret = []
        for env in self.envs:
            state, obs = env.reset(max_step=max_step)
            reward = dict.fromkeys(obs.keys(), 0.0)
            dones = dict.fromkeys(obs.keys(), False)
            dones["__all__"] = False
            ret.append((state, obs, reward, dones))

        return ret

    def step(
        self, actions: Dict[AgentID, np.ndarray]
    ) -> List[Tuple["states", "observations", "rewards", "dones", "infos"]]:
        """Environment stepping function.

        Args:
            actions (Dict[EnvID, Dict[AgentID, Any]]): A dict of action dict, one for an environment.

        Returns:
            Dict[EnvID, Sequence[Dict[AgentID, Any]]]: A dict of environment returns.
        """

        env_rets = []

        for i, env in enumerate(self.envs):
            _actions = {k: v[i] for k, v in actions.items()}
            state, obs, rew, done, info = env.step(_actions)
            env_done = done["__all__"]
            self.step_cnt += 1

            if self.is_terminated():
                env_done = True

            if env_done:
                # replace ret with the new started obs
                self.cached_episode_infos.append(env.collect_info())
                state, obs = env.reset(max_step=self.max_step)
            env_rets.append((state, obs, rew, done))
        return env_rets

    def is_terminated(self):
        if isinstance(self.step_cnt, int):
            return self.step_cnt >= self.fragment_length
        else:
            raise NotImplementedError
            # return self._step_cnt[self._trainable_agents] >= self._fragment_length

    def action_adapter(
        self, policy_outputs: List[Dict[str, Dict[AgentID, Any]]]
    ) -> List[Dict[AgentID, Any]]:

        # since activ_envs maybe updated after self.step, so we should use keys
        # in self.active_envs
        res = defaultdict(list)
        for e in policy_outputs:
            for agent, v in e.items():
                res[agent].append(v[Episode.ACTION].squeeze())
        # them stack
        res = dict(res)
        for k, v in res.items():
            res[k] = np.stack(v)
        return res

    def collect_info(self) -> Dict[EnvID, Dict[str, Any]]:
        """Collect information of each episode since last running.

        Returns:
            Dict[EnvID, Dict[str, Any]]: A dict of episode informations, mapping from environment ids to dicts.
        """

        ret = self.cached_episode_infos.copy()
        return ret

    def close(self):
        for env in self._envs:
            env.close()


@ray.remote(num_cpus=0)
class _RemoteEnv:
    def __init__(self, creater: Callable, env_config: Dict[str, Any]) -> None:
        self.env: Environment = creater(**env_config)
        self.runtime_id = None

    def reset(self, runtime_id: str, **kwargs) -> Dict[str, Dict[AgentID, Any]]:
        self.runtime_id = runtime_id
        ret = self.env.reset(**kwargs)
        return {self.runtime_id: ret}

    def step(self, action: Dict[AgentID, Any]):
        ret = self.env.step(action)
        return {self.runtime_id: ret}

    def action_adapter(self, policy_outpus):
        return {self.runtime_id: self.env.action_adapter(policy_outpus)}

    def collect_info(self):
        if self.env.cnt > 0:
            return {self.runtime_id: self.env.collect_info()}
        else:
            return {}

    def from_env(self, env):
        assert isinstance(env, Environment)
        self.env = env
        return self


# pragma: no cover
class SubprocVecEnv(VectorEnv):
    pass
