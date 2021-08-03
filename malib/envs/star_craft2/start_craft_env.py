import numpy as np

from gym import spaces
from pettingzoo import ParallelEnv
from smac.env import StarCraft2Env as sc_env

from malib.envs import Environment
from malib.utils.typing import Dict, AgentID, Any
from malib.backend.datapool.offline_dataset_server import Episode


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


class _SC2Env(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        super(_SC2Env, self).__init__()
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
        n_state = env_info["state_shape"]
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
        self.state_space = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(n_state,), dtype=np.int8
        )

        self.action_spaces = dict(
            zip(
                self.possible_agents,
                [spaces.Discrete(num_actions) for _ in range(self.n_agents)],
            )
        )
        self.env_info = env_info

    @property
    def global_state_space(self):
        return self.state_space

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
        return (
            {aid: state for aid in obs},
            obs,
            {aid: _obs["action_mask"] for aid, _obs in obs.items()},
        )

    def step(self, actions):
        act_list = [actions[aid] for aid in self.agents]
        reward, terminated, info = self.smac_env.step(act_list)
        next_state = self.get_state()
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
        return (
            {aid: next_state for aid in next_obs_dict},
            next_obs_dict,
            {aid: _next_obs["action_mask"] for aid, _next_obs in next_obs_dict.items()},
            rew_dict,
            done_dict,
            info_dict,
        )

    def get_state(self):
        return self.smac_env.get_state()

    def render(self, mode="human"):
        """not implemented now in smac"""
        pass

    def close(self):
        self.smac_env.close()


class SC2Env(Environment):
    def __init__(self, **configs):
        super(SC2Env, self).__init__(**configs)

        self.is_sequential = False
        self._env_id = configs["env_id"]
        self._env = _SC2Env(**configs["scenario_configs"])
        self._trainable_agents = self._env.possible_agents

        # register extra returns
        self._extra_returns = [
            Episode.NEXT_STATE,
            Episode.CUR_STATE,
            Episode.ACTION_MASK,
            "next_action_mask",
        ]

    @property
    def env_info(self):
        return self._env.env_info

    @property
    def global_state_space(self):
        return self._env.global_state_space

    def step(self, actions: Dict[AgentID, Any]):
        states, observations, action_masks, rewards, dones, infos = self._env.step(
            actions
        )
        return {
            Episode.NEXT_STATE: states,
            Episode.NEXT_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            # Episode.INFO: infos,
            "next_action_mask": action_masks,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self):
        states, observations, action_masks = self._env.reset()
        return {
            Episode.CUR_STATE: states,
            Episode.CUR_OBS: observations,
            Episode.ACTION_MASK: action_masks,
        }


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
