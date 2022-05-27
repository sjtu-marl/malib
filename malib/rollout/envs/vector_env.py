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

from collections import ChainMap
from typing import Tuple, Dict, Any, List, Callable

import uuid

import gym
import ray

from ray.actor import ActorHandle

from malib.utils.logging import Logger
from malib.utils.typing import (
    AgentID,
    EnvID,
    PolicyID,
)
from malib.rollout.envs import Environment
from malib.utils.episode import Episode


class VectorEnv:
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        preset_num_envs: int = 0,
    ):
        """Create a VectorEnv instance.

        :param Dict[AgentID,gym.Space] observation_spaces: A dict of agent observation space
        :param Dict[AgentID,gym.Space] action_spaces: A dict of agent action space
        :param type creator: The handler to create environment
        :param Dict[str,Any] configs: Environment configuration
        :param int num_envs: The number of nested environment
        :param int fragment_length: The maximum of batched frames

        """
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.possible_agents = list(observation_spaces.keys())

        self._num_envs = preset_num_envs
        self._creator = creator
        self._configs = configs.copy()
        self._envs: List[Environment] = []
        self._step_cnt = 0
        self._limits = len(self._envs)
        self._fragment_length = None
        self._active_envs = {}
        self._cached_episode_infos = {}
        self._action_adapter = creator.action_adapter

        self.add_envs(num=preset_num_envs)

    @property
    def batched_step_cnt(self) -> int:
        # return (
        #     self._step_cnt
        #     if isinstance(self._step_cnt, int)
        #     else self._step_cnt[self._trainable_agents]
        # )
        # XXX(ming): we currently considers only int case
        return self._step_cnt

    @property
    def num_envs(self) -> int:
        """The total number of environments"""

        return self._num_envs

    @property
    def envs(self) -> List[Environment]:
        """Return a limited list of enviroments"""

        return self._envs[: self._limits]

    @property
    def env_creator(self):
        return self._creator

    @property
    def env_configs(self):
        return self._configs

    @property
    def limits(self):
        return self._limits

    @property
    def active_envs(self) -> Dict[EnvID, Environment]:
        return self._active_envs

    @classmethod
    def from_envs(cls, envs: List[Environment], config: Dict[str, Any]):
        """Generate vectorization environment from exisiting environments."""

        observation_spaces = envs[0].observation_spaces
        action_spaces = envs[0].action_spaces

        vec_env = cls(observation_spaces, action_spaces, type(envs[0]), config, 0)
        vec_env.add_envs(envs=envs)

        return vec_env

    def add_envs(self, envs: List = None, num: int = 0):
        """Add exisiting `envs` or `num` new environments to this vectorization environment.
        If `envs` is not empty or None, the `num` will be ignored.
        """

        if envs and len(envs) > 0:
            for env in envs:
                self._envs.append(env)
                self._num_envs += 1
            Logger.debug(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            for _ in range(num):
                self._envs.append(self.env_creator(**self.env_configs))
                self._num_envs += 1
            Logger.debug(f"created {num} new environments.")

    def reset(
        self,
        fragment_length: int,
        max_step: int,
        custom_reset_config: Dict[str, Any] = None,
        trainable_mapping: Dict[AgentID, PolicyID] = None,
    ) -> Dict[EnvID, Dict[str, Dict[AgentID, Any]]]:
        self._limits = self.num_envs
        self._step_cnt = 0
        self._fragment_length = fragment_length
        self._custom_reset_config = custom_reset_config
        self.max_step = max_step
        self._cached_episode_infos = {}

        # generate runtime env ids
        runtime_ids = [uuid.uuid1().hex for _ in range(self._limits)]
        self._active_envs = dict(zip(runtime_ids, self.envs[: self._limits]))

        self._trainable_agents = (
            list(trainable_mapping.keys()) if trainable_mapping is not None else None
        )

        ret = {}
        for env_id, env in self.active_envs.items():
            _ret = env.reset(max_step=max_step, custom_reset_config=custom_reset_config)
            if isinstance(_ret, Dict):
                _ret = (_ret,)
            ret[env_id] = _ret

        return ret

    def step(self, actions: Dict[EnvID, Dict[AgentID, Any]]) -> Dict:
        active_envs = self.active_envs

        env_rets = {}
        dead_envs = []
        # FIXME(ming): (keyerror, sometimes) the env_id in actions is not an active environment.
        for env_id, _actions in actions.items():
            ret = active_envs[env_id].step(_actions)
            env_done = ret[3]["__all__"]
            env = self.active_envs[env_id]
            self._update_step_cnt()
            if env_done:
                env = active_envs.pop(env_id)
                dead_envs.append(env)
                self._cached_episode_infos[env_id] = env.collect_info()
            env_rets[env_id] = ret

        if not self.is_terminated() and len(dead_envs) > 0:
            for env in dead_envs:
                _tmp = env.reset(
                    max_step=self.max_step,
                    custom_reset_config=self._custom_reset_config,
                )
                # regenerate runtime id
                runtime_id = uuid.uuid1().hex
                self._active_envs[runtime_id] = env
                env_rets[runtime_id] = _tmp
        return env_rets

    def _update_step_cnt(self, ava_agents: List[AgentID] = None):
        if isinstance(self._step_cnt, int):
            self._step_cnt += 1
        else:
            raise NotImplementedError

    def is_terminated(self):
        if isinstance(self._step_cnt, int):
            return self._step_cnt >= self._fragment_length
        else:
            raise NotImplementedError
            # return self._step_cnt[self._trainable_agents] >= self._fragment_length

    def action_adapter(
        self, policy_outputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]]
    ) -> Dict[EnvID, Dict[AgentID, Any]]:

        # since activ_envs maybe updated after self.step, so we should use keys
        # in self.active_envs
        res = {
            env_id: {agent: v["action"][0] for agent, v in policy_output.items()}
            for env_id, policy_output in policy_outputs.items()
        }

        return res

    def collect_info(self, truncated=False) -> List[Dict[str, Any]]:
        # XXX(ziyu): We can add a new param 'truncated' to determine whether to add
        # the nonterminal env_info into rets.
        ret = self._cached_episode_infos
        for runtime_id, env in self.active_envs.items():
            if env.cnt > 0 and (runtime_id not in ret):
                ret[runtime_id] = env.collect_info()

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


class SubprocVecEnv(VectorEnv):
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
    ):
        # modify creator as remote creator
        self.pending_tasks = []

        super(SubprocVecEnv, self).__init__(
            observation_spaces, action_spaces, creator, configs
        )

    def action_adapter(
        self, policy_outputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]]
    ) -> Dict[EnvID, Dict[AgentID, Any]]:
        res = {}

        # since activ_envs maybe updated after self.step, so we should use keys
        # in self.active_envs
        res = ray.get(
            [
                self.active_envs[env_id].action_adapter.remote(policy_output)
                for env_id, policy_output in policy_outputs.items()
            ]
        )
        res = ChainMap(*res)

        return res

    def add_envs(self, envs: List = None, num: int = 0):
        """Add existin envs to remote actor"""

        # num_env_pool = self.num_envs

        if envs and len(envs) > 0:
            for env in envs:
                if isinstance(env, ActorHandle):
                    self._envs.append(env)
                else:
                    self._envs.append(ray.get(_RemoteEnv.from_env(env).remote()))

                self._num_envs += 1
            Logger.debug(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            # num = min(
            #     self.max_num_envs - self.num_envs, max(0, self.max_num_envs - num)
            # )
            for _ in range(num):
                self._envs.append(
                    _RemoteEnv.remote(
                        creater=self.env_creator, env_config=self.env_configs
                    )
                )
                self._num_envs += 1
            Logger.info(f"created {num} new environments.")

    def reset(
        self,
        limits: int,
        fragment_length: int,
        max_step: int,
        custom_reset_config: Dict[str, Any] = None,
        trainable_mapping: Dict[AgentID, PolicyID] = None,
    ) -> Dict[EnvID, Dict[str, Dict[AgentID, Any]]]:
        self._limits = limits or self.num_envs
        self._step_cnt = 0
        self._fragment_length = fragment_length
        self._custom_reset_config = custom_reset_config
        self.max_step = max_step

        # clean peanding tasks
        if len(self.pending_tasks) > 0:
            _ = ray.get(self.pending_tasks)

        # generate runtime env ids
        runtime_ids = [uuid.uuid1().hex for _ in range(self._limits)]
        self._active_envs = dict(zip(runtime_ids, self.envs[: self._limits]))

        self._trainable_agents = (
            list(trainable_mapping.keys()) if trainable_mapping is not None else None
        )

        self._step_cnt = 0

        ret = {}

        for env_id, env in self.active_envs.items():
            _ret = ray.get(
                env.reset.remote(
                    runtime_id=env_id,
                    max_step=max_step,
                    custom_reset_config=custom_reset_config,
                )
            )
            ret.update(_ret)

        ret = ray.get(
            [
                env.reset.remote(
                    runtime_id=runtime_id,
                    max_step=max_step,
                    custom_reset_config=custom_reset_config,
                )
                for runtime_id, env in self._active_envs.items()
            ]
        )
        ret = ChainMap(*ret)

        return ret

    def step(self, actions: Dict[AgentID, List]) -> Dict:

        for env_id, _actions in actions.items():
            self.pending_tasks.append(self.active_envs[env_id].step.remote(_actions))

        ready_tasks = []
        # XXX(ming): should raise a warning, if too many retries.
        while len(ready_tasks) < 1:
            ready_tasks, self.pending_tasks = ray.wait(
                self.pending_tasks, num_returns=len(self.pending_tasks), timeout=0.5
            )
        rets = dict(ChainMap(*ray.get(ready_tasks)))
        self._step_cnt += len(rets)

        # FIXME(ming): (runtime error, sometimes) dictionary changed size during iteration
        dead_envs = []
        for env_id, _ret in rets.items():
            dones = _ret[Episode.DONE]
            env_done = any(dones.values())
            if env_done:
                # reset and assign a new runtime
                env = self._active_envs.pop(env_id)
                runtime_id = uuid.uuid1().hex
                self.active_envs[runtime_id] = env
                dead_envs.append(env)

        if not self.is_terminated() and len(dead_envs) > 0:
            for env in dead_envs:
                rets.update(
                    ray.get(
                        env.reset.remote(
                            runtime_id=runtime_id,
                            max_step=self.max_step,
                            custom_reset_config=self._custom_reset_config,
                        )
                    )
                )

        return rets

    def collect_info(self, truncated=False) -> List[Dict[str, Any]]:
        # XXX(ziyu): We can add a new param 'truncated' to determine whether to add
        # the nonterminal env_info into rets.
        ret = self._cached_episode_infos
        additional = ray.get(
            [env.collect_info.remote() for env in self.active_envs.values()]
        )
        additional = ChainMap(*additional)
        ret.update(additional)
        return ret

    def close(self):
        for env in self.envs:
            # assert isinstance(env, ActorHandle)
            env.__ray_terminate__.remote()
