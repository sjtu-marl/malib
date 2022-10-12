# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

from typing import ChainMap, Dict, Any, List, Union, Sequence
from collections import defaultdict
from readerwriterlock import rwlock

import gym
import uuid
import ray

from malib.utils.typing import AgentID, EnvID
from malib.utils.episode import Episode
from .vector_env import VectorEnv, SubprocVecEnv


class AsyncVectorEnv(VectorEnv):
    """Compared to VectorEnv, AsyncVectorEnv has some differences in the environment stepping. An AsyncVectorEnv accepts a dict \
        of agent actions, but the environment return is determined by 'whether all actions of the current step have been collected'. \
            In other words, an AsyncVectorEnv performs environment stepping only when all alive agents are ready. \
                None will be returned when a sub-environment does not ready yet.
    """

    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        preset_num_envs: int = 0,
    ):
        super().__init__(
            observation_spaces, action_spaces, creator, configs, preset_num_envs
        )
        self.cached_frame = defaultdict(lambda: {})

    def _delay_step(
        self, env_id: EnvID, actions: Dict[AgentID, Any]
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        """Performs delayed environment stepping. Accepts an agent action dict relates to an environment.

        Args:
            env_id (EnvID): Environment key.
            actions (Dict[AgentID, Any]): Agent action dict.

        Returns:
            Union[None, Dict[str, Dict[AgentID, Any]]]: When all agents are ready for the environment stepping, then return the environment returns, otherwise None.
        """

        rets = None
        if env_id in self.active_envs:
            cached_frame = self.cached_frame[env_id]
            for aid, action in actions.items():
                if aid not in cached_frame:
                    cached_frame[aid] = action

            if len(cached_frame) == len(self.possible_agents):
                rets = self.active_envs[env_id].step(cached_frame)
                self.cached_frame[env_id] = {}

        return rets

    def step(
        self, actions: Dict[EnvID, Dict[AgentID, Any]]
    ) -> Dict[EnvID, Sequence[Dict[AgentID, Any]]]:
        """Asynchrounous stepping, maybe return an empty dict of environment returns."""
        env_rets = {}
        dead_envs = []

        for env_id, _actions in actions.items():
            ret = self._delay_step(env_id, _actions)
            if ret is None:
                continue
            env_done = ret[3]["__all__"]
            env = self.active_envs[env_id]

            self._update_step_cnt()

            if env_done:
                env = self.active_envs.pop(env_id)
                dead_envs.append(env)
                self._cached_episode_infos[env_id] = env.collect_info()
                # the terminate judgement is wrong, since step cnt count is delayed
            env_rets[env_id] = ret

        if not self.is_terminated() and len(dead_envs) > 0:
            for env in dead_envs:
                _tmp = env.reset(max_step=self.max_step)
                runtime_id = uuid.uuid1().hex
                self._active_envs[runtime_id] = env
                env_rets[runtime_id] = _tmp

        return env_rets


class AsyncSubProcVecEnv(SubprocVecEnv):
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        preset_num_envs: int = 0,
    ):
        super().__init__(observation_spaces, action_spaces, creator, configs)
        self.num_envs = preset_num_envs
        self.cached_frame = defaultdict(lambda: {})

    def _delay_step(
        self, env_id: EnvID, actions: Dict[AgentID, Any]
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        rets = None
        cached_frame = self.cached_frame[env_id]
        for aid, action in actions.items():
            if aid not in cached_frame:
                cached_frame[aid] = action

        if len(cached_frame) == len(self.possible_agents):
            rets = self.active_envs[env_id].step.remote(cached_frame)
            self.cached_frame[env_id] = {}

        return rets

    def step(
        self, actions: Dict[EnvID, Dict[AgentID, Any]]
    ) -> Dict[EnvID, Sequence[Dict[AgentID, Any]]]:
        env_rets = {}
        dead_envs = []

        for env_id, _actions in actions.items():
            task = self._delay_step(env_id, _actions)
            if task is None:
                continue
            self.pending_tasks.append(task)

        rets = ray.get(self.pending_tasks)
        rets = ChainMap(*rets)
        for env_id, ret in rets.items():
            env_done = ret[Episode.DONE]["__all__"]
            env = self.active_envs[env_id]

            self._update_step_cnt()

            if env_done:
                env = self.active_envs.pop(env_id)
                dead_envs.append(env)
                self._cached_episode_infos[env_id] = env.collect_info()
                # the terminate judgement is wrong, since step cnt count is delayed
            env_rets[env_id] = ret

        return env_rets
