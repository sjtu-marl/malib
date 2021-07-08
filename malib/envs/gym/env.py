import gym
from pettingzoo import ParallelEnv

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode


class _GymEnv(ParallelEnv):
    def __init__(self, env_name, **kwargs):
        super(_GymEnv, self).__init__(**kwargs)
        self.gym_env = gym.make(env_name, **kwargs)
        self.default_agent_name = "agent_0"

        self.is_sequential = False
        self.possible_agents = [self.default_agent_name]
        self.observation_spaces = {self.default_agent_name: self.gym_env.observation_space}
        self.action_spaces = {self.default_agent_name: self.gym_env.action_space}

    def seed(self, seed=None):
        self.gym_env.seed(seed)

    def reset(self):
        obs_t = self.gym_env.reset()
        obs = {self.default_agent_name: obs_t}
        return obs

    def step(self, actions):
        action = actions[self.default_agent_name]
        next_obs, reward, done, info = self.gym_env.step(action)
        next_obs_dict = {self.default_agent_name: next_obs}
        reward_dict = {self.default_agent_name: reward}
        done_dict = {self.default_agent_name: done}
        info_dict = {self.default_agent_name: info}
        return (
            next_obs_dict,
            reward_dict,
            done_dict,
            info_dict,
        )

    def close(self):
        self.gym_env.close()


class GymEnv(Environment):
    def __init__(self, **configs):
        super(GymEnv, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})

        self.is_sequential = False
        self._env = _GymEnv(env_id, **scenario_configs)
        self._trainable_agents = self._env.possible_agents

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(actions)
        return {
            Episode.NEXT_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            Episode.INFO: infos,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self):
        observations = self._env.reset()
        return {Episode.CUR_OBS: observations}
