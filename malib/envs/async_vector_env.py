from collections import defaultdict
from typing import ChainMap
import gym
import uuid
import ray

from malib.utils.typing import Dict, AgentID, Any, List, EnvID, Union
from malib.utils.episode import EpisodeKey
from .vector_env import VectorEnv, SubprocVecEnv


class AsyncVectorEnv(VectorEnv):
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

    def step(self, actions: Dict[EnvID, Dict[AgentID, Any]]) -> Dict:
        """Execute environment step. Note the enter environment ids could be different from the output.

        :param actions: A dict of action dict.
        :type actions: Dict[EnvID, Dict[AgentID, Any]]
        :return: A dict of environment return dict.
        :rtype: Dict
        """

        env_rets = {}
        dead_envs = []

        for env_id, _actions in actions.items():
            ret = self._delay_step(env_id, _actions)
            if ret is None:
                continue
            env_done = ret[EpisodeKey.DONE]["__all__"]
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
                _tmp = env.reset(
                    max_step=self.max_step,
                    custom_reset_config=self._custom_reset_config,
                )
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
        """Submit async remote task.

        :param env_id: _description_
        :type env_id: EnvID
        :param actions: _description_
        :type actions: Dict[AgentID, Any]
        :return: _description_
        :rtype: Union[None, Dict[str, Dict[AgentID, Any]]]
        """

        rets = None
        cached_frame = self.cached_frame[env_id]
        for aid, action in actions.items():
            if aid not in cached_frame:
                cached_frame[aid] = action

        if len(cached_frame) == len(self.possible_agents):
            rets = self.active_envs[env_id].step.remote(cached_frame)
            self.cached_frame[env_id] = {}

        return rets

    def step(self, actions: Dict[EnvID, Dict[AgentID, Any]]) -> Dict:
        """Execute environment step. Note the enter environment ids could be different from the output.

        :param actions: A dict of action dict.
        :type actions: Dict[EnvID, Dict[AgentID, Any]]
        :return: A dict of environment return dict.
        :rtype: Dict
        """

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
            env_done = ret[EpisodeKey.DONE]["__all__"]
            env = self.active_envs[env_id]

            self._update_step_cnt()

            if env_done:
                env = self.active_envs.pop(env_id)
                dead_envs.append(env)
                self._cached_episode_infos[env_id] = env.collect_info()
                # the terminate judgement is wrong, since step cnt count is delayed
            env_rets[env_id] = ret

        return env_rets
