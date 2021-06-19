# -*- coding: utf-8 -*-
import numpy as np
from gym import spaces
from pettingzoo import ParallelEnv
from smac.env import StarCraft2Env as sc_env

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


class SC2Env(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        super(SC2Env, self).__init__()
        self.smac_env = sc_env(**kwargs)
        env_info = self.smac_env.get_env_info()
        self.kwargs = kwargs
        self.n_agents = self.smac_env.n_agents

        self.possible_agents = agents_list.get(
            self.kwargs["map_name"], [f"{i}" for i in range(self.n_agents)]
        )
        self.agents = []

        n_obs = env_info["obs_shape"]
        num_actions = env_info["n_actions"]
        self.observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    spaces.Dict(
                        {
                            "observation": spaces.Box(
                                low=0.0, high=1.0, shape=(n_obs,), dtype=np.int8
                            ),
                            "action_mask": spaces.Box(
                                low=0, high=1, shape=(num_actions,), dtype=np.int8
                            ),
                        }
                    )
                    for _ in range(self.n_agents)
                ],
            )
        )

        self.action_spaces = dict(
            zip(
                self.possible_agents,
                [spaces.Discrete(num_actions) for _ in range(self.n_agents)],
            )
        )

    def seed(self, seed=None):
        self.smac_env = sc_env(**self.kwargs)
        self.agents = []

    def reset(self):
        """only return observation not return state"""
        self.agents = self.possible_agents
        obs_t, state = self.smac_env.reset()
        action_mask = self.smac_env.get_avail_actions()
        obs = {
            aid: {"observation": obs_t[i], "action_mask": np.array(action_mask[i])}
            for i, aid in enumerate(self.agents)
        }
        return obs

    def step(self, actions):
        act_list = [actions[aid] for aid in self.agents]
        reward, terminated, info = self.smac_env.step(act_list)
        next_obs_t = self.smac_env.get_obs()
        next_action_mask = self.smac_env.get_avail_actions()
        rew_dict = {agent_name: reward for agent_name in self.agents}
        done_dict = {agent_name: terminated for agent_name in self.agents}
        next_obs_dict = {
            aid: {
                "observation": next_obs_t[i],
                "action_mask": np.array(next_action_mask[i]),
            }
            for i, aid in enumerate(self.agents)
        }

        info_dict = {
            aid: {**info, "action_mask": next_action_mask[i]}
            for i, aid in enumerate(self.agents)
        }
        if terminated:
            self.agents = []
        return next_obs_dict, rew_dict, done_dict, info_dict

    def get_state(self):
        return self.smac_env.get_state()

    def render(self, mode="human"):
        """not implemented now in smac"""
        # self._env.render()
        pass

    def close(self):
        # XXX(ming): avoid redundant close
        self.smac_env.close()


if __name__ == "__main__":
    env_config = {"map_name": "3m"}
    env = SC2Env(**env_config)
    print(env.possible_agents)

    for aid, obsp in env.observation_spaces.items():
        print(aid, type(obsp))

    obs = env.reset()

    while True:
        act_dict = {}
        for i, aid in enumerate(env.agents):
            legal_act = np.nonzero(obs[aid]["action_mask"])[0]
            act_dict[aid] = np.random.choice(legal_act, 1)
        print(act_dict)
        print(obs)
        next_obs, rew, done, info = env.step(act_dict)
        print(rew, done)
        print(info)
        obs = next_obs
        if all(done.values()):
            break
        print()
