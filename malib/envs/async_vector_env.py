from collections import defaultdict
import gym
import uuid

from malib.utils.typing import Dict, AgentID, Any, List, EnvID, Union
from malib.utils.episode import EpisodeKey
from .vector_env import VectorEnv


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

        for env_id, _actions in actions.items():
            ret = self._delay_step(env_id, _actions)
            if ret is None:
                continue
            env_done = ret[EpisodeKey.DONE]["__all__"]
            env = self.active_envs[env_id]
            if env_done:
                env = self.active_envs.pop(env_id)
                self._cached_episode_infos[env_id] = env.collect_info()
                if not self.is_terminated():
                    _tmp = env.reset(
                        max_step=self.max_step,
                        custom_reset_config=self._custom_reset_config,
                    )
                    ret.update(_tmp)
                    # regenerate runtime id
                    runtime_id = uuid.uuid1().hex
                    self._active_envs[runtime_id] = env
                    env_rets[runtime_id] = _tmp
            env_rets[env_id] = ret
        if isinstance(self._step_cnt, int):
            self._step_cnt += len(env_rets)
        else:
            for _actions in actions.values():
                if self._trainable_agents in _actions:
                    self._step_cnt[self._trainable_agents] += 1
        return env_rets
