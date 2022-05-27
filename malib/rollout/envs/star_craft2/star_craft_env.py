from typing import Dict, Any, List, Union

import numpy as np
import gym

from gym import spaces
from smac.env import StarCraft2Env as sc_env

from malib.rollout.envs import Environment
from malib.rollout.envs.env import GroupWrapper
from malib.utils.typing import AgentID
from malib.utils.episode import Episode


agents_list = {
    "3m": [f"Marine_{i}" for i in range(3)],
    "8m": [f"Marine_{i}" for i in range(8)],
    "25m": [f"Marine_{i}" for i in range(25)],
    "2s3z": [f"Stalkers_{i}" for i in range(2)] + [f"Zealots_{i}" for i in range(3)],
    "2s5z": [f"Stalkers_{i}" for i in range(2)] + [f"Zealots_{i}" for i in range(3)],
}
# FIXME(ziyu): better ways or complete the rest map information


def get_agent_names(map_name):
    if map_name in agents_list:
        return agents_list[map_name]
    else:
        return None


# class _SC2Env(ParallelEnv):
#     metadata = {"render.modes": ["human"]}

#     def __init__(self, **kwargs):
#         super(_SC2Env, self).__init__()
#         self.smac_env = sc_env(**kwargs)
#         env_info = self.smac_env.get_env_info()
#         self.kwargs = kwargs
#         self.n_agents = self.smac_env.n_agents

#         self.possible_agents = agents_list.get(
#             self.kwargs["map_name"], [f"{i}" for i in range(self.n_agents)]
#         )
#         self.agents = []

#         n_obs = env_info["obs_shape"]
#         num_actions = env_info["n_actions"]
#         n_state = env_info["state_shape"]
#         self.observation_spaces = dict(
#             zip(
#                 self.possible_agents,
#                 [
#                     spaces.Dict(
#                         {
#                             "observation": spaces.Box(
#                                 low=0.0, high=1.0, shape=(n_obs,), dtype=np.int8
#                             ),
#                             "action_mask": spaces.Box(
#                                 low=0, high=1, shape=(num_actions,), dtype=np.int8
#                             ),
#                         }
#                     )
#                     for _ in range(self.n_agents)
#                 ],
#             )
#         )
#         self.state_space = spaces.Box(
#             low=-np.inf, high=+np.inf, shape=(n_state,), dtype=np.int8
#         )

#         self.action_spaces = dict(
#             zip(
#                 self.possible_agents,
#                 [spaces.Discrete(num_actions) for _ in range(self.n_agents)],
#             )
#         )
#         self.env_info = env_info

#     @property
#     def global_state_space(self):
#         return self.state_space

#     def seed(self, seed=None):
#         self.smac_env = sc_env(**self.kwargs)
#         self.agents = []

#     def reset(self):
#         """only return observation not return state"""
#         self.agents = self.possible_agents
#         obs_t, state = self.smac_env.reset()
#         action_mask = self.smac_env.get_avail_actions()
#         obs = {
#             aid: {"observation": obs_t[i], "action_mask": np.array(action_mask[i])}
#             for i, aid in enumerate(self.agents)
#         }
#         return (
#             {aid: state for aid in obs},
#             obs,
#             {aid: _obs["action_mask"] for aid, _obs in obs.items()},
#         )

#     def step(self, actions):
#         act_list = [actions[aid] for aid in self.agents]
#         reward, terminated, info = self.smac_env.step(act_list)
#         next_state = self.get_state()
#         next_obs_t = self.smac_env.get_obs()
#         next_action_mask = self.smac_env.get_avail_actions()
#         rew_dict = {agent_name: reward for agent_name in self.agents}
#         done_dict = {agent_name: terminated for agent_name in self.agents}
#         next_obs_dict = {
#             aid: {
#                 "observation": next_obs_t[i],
#                 "action_mask": np.array(next_action_mask[i]),
#             }
#             for i, aid in enumerate(self.agents)
#         }

#         info_dict = {
#             aid: {**info, "action_mask": next_action_mask[i]}
#             for i, aid in enumerate(self.agents)
#         }
#         if terminated:
#             self.agents = []
#         return (
#             {aid: next_state for aid in next_obs_dict},
#             next_obs_dict,
#             {aid: _next_obs["action_mask"] for aid, _next_obs in next_obs_dict.items()},
#             rew_dict,
#             done_dict,
#             info_dict,
#         )

#     def get_state(self):
#         return self.smac_env.get_state()

#     def render(self, mode="human"):
#         """not implemented now in smac"""
#         pass

#     def close(self):
#         self.smac_env.close()


class SC2Env(Environment):
    def __init__(self, **configs):
        super(SC2Env, self).__init__(**configs)

        env = sc_env(**configs["scenario_configs"])
        env_info = env.get_env_info()

        n_obs = env_info["obs_shape"]
        num_actions = env_info["n_actions"]
        n_state = env_info["state_shape"]

        self.is_sequential = False
        self.max_step = 1000
        self.env_info = env_info
        self.scenario_configs = configs["scenario_configs"]

        self._env = env
        self._possible_agents = get_agent_names(self.scenario_configs["map_name"])

        n_agents = len(self.possible_agents)

        self._observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    spaces.Dict(
                        {
                            "observation": spaces.Box(
                                low=-1.0, high=1.0, shape=(n_obs,), dtype=np.float32
                            ),
                            "action_mask": spaces.Box(
                                low=0, high=1, shape=(num_actions,), dtype=int
                            ),
                        }
                    )
                    for _ in range(n_agents)
                ],
            )
        )

        self._action_spaces = dict(
            zip(
                self.possible_agents,
                [spaces.Discrete(num_actions) for _ in range(n_agents)],
            )
        )

        self._trainable_agents = self.possible_agents

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._possible_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def get_state(self):
        return self._env.get_state()

    def seed(self, seed: int = None):
        pass

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super(SC2Env, self).reset(
            max_step=max_step, custom_reset_config=custom_reset_config
        )

        obs_t, state = self._env.reset()
        action_mask = self._env.get_avail_actions()
        obs = {
            aid: {"observation": obs_t[i], "action_mask": np.array(action_mask[i])}
            for i, aid in enumerate(self.possible_agents)
        }
        return {
            Episode.CUR_OBS: obs,
            Episode.ACTION_MASK: {
                aid: _obs["action_mask"] for aid, _obs in obs.items()
            },
        }

    def time_step(self, actions: Dict[AgentID, Any]):
        act_list = [actions[aid] for aid in self.possible_agents]
        reward, terminated, info = self._env.step(act_list)
        # next_state = self.get_state()
        next_obs_t = self._env.get_obs()
        next_action_mask = self._env.get_avail_actions()
        rew_dict = {agent_name: reward for agent_name in self.possible_agents}
        done_dict = {agent_name: terminated for agent_name in self.possible_agents}

        next_obs_dict = {
            aid: {
                "observation": next_obs_t[i],
                "action_mask": np.array(next_action_mask[i]),
            }
            for i, aid in enumerate(self.possible_agents)
        }

        info_dict = {
            aid: {**info, "action_mask": next_action_mask[i]}
            for i, aid in enumerate(self.possible_agents)
        }

        return {
            # {aid: next_state for aid in next_obs_dict},
            Episode.NEXT_OBS: next_obs_dict,
            Episode.ACTION_MASK: {
                aid: _next_obs["action_mask"]
                for aid, _next_obs in next_obs_dict.items()
            },
            Episode.REWARD: rew_dict,
            Episode.DONE: done_dict,
            # Episode.INFO: info_dict,
        }

    def close(self):
        self._env.close()


def StatedSC2(**config):

    env = SC2Env(**config)

    class Wrapped(GroupWrapper):
        def __init__(self, env: Environment):
            super(Wrapped, self).__init__(env)

        def build_state_spaces(self) -> Dict[AgentID, gym.Space]:
            return {
                agent: spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.env.env_info["state_shape"],),
                )
                for agent in self.possible_agents
            }

        def build_state_from_observation(
            self, agent_observation: Dict[AgentID, Any]
        ) -> Dict[AgentID, Any]:
            state = self.env.get_state()
            return dict.fromkeys(self.possible_agents, state)

        def group_rule(self, agent_id: AgentID) -> str:
            return agent_id.split("_")[0]

    return Wrapped(env)
