from typing import Dict, Any, List, Union

import numpy as np
import gym

from gym import spaces
from smac.env import StarCraft2Env as sc_env

from malib.rollout.envs.env import Environment, GroupWrapper
from malib.utils.typing import AgentID


agents_list = {
    "3m": [f"Marine_{i}" for i in range(3)],
    "8m": [f"Marine_{i}" for i in range(8)],
    "25m": [f"Marine_{i}" for i in range(25)],
    "2s3z": [f"Stalkers_{i}" for i in range(2)] + [f"Zealots_{i}" for i in range(3)],
    "2s5z": [f"Stalkers_{i}" for i in range(2)] + [f"Zealots_{i}" for i in range(3)],
}


def get_agent_names(map_name):
    if map_name in agents_list:
        return agents_list[map_name]
    else:
        return None


class SC2Env(Environment):
    def __init__(self, **configs):
        super(SC2Env, self).__init__(**configs)

        env = sc_env(map_name=configs["env_id"])
        env_info = env.get_env_info()

        n_obs = env_info["obs_shape"]
        num_actions = env_info["n_actions"]
        n_state = env_info["state_shape"]

        self.max_step = 1000
        self.env_info = env_info
        self.scenario_configs = configs.get("scenario_configs", {})

        self._env = env
        self._possible_agents = get_agent_names(configs["env_id"])

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

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._possible_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def get_state(self) -> np.ndarray:
        return self._env.get_state()

    def seed(self, seed: int = None):
        """Modify the default seed of underlying environment.

        Args:
            seed (int, optional): Seed. Defaults to None.
        """
        self._env._seed = seed

    def reset(self, max_step: int = None) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super(SC2Env, self).reset()

        obs_t, state = self._env.reset()
        action_mask = self._env.get_avail_actions()
        obs = {
            aid: {"observation": obs_t[i], "action_mask": np.array(action_mask[i])}
            for i, aid in enumerate(self.possible_agents)
        }

        # convert state to a dict of state, in agent-wise
        agent_state = dict.fromkeys(self.possible_agents, state)
        return agent_state, obs

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

        state_dict = dict.fromkeys(self.possible_agents, self.get_state())

        return (state_dict, next_obs_dict, rew_dict, done_dict, info_dict)

    def close(self):
        self._env.close()

        try:
            import subprocess

            subprocess.run(
                ["ps", "-ef|grep StarCraft|grep -v grep|cut -c 9-15|xargs kill -9"]
            )
        except Exception as e:
            print("[warning]: {}".format(e))


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


if __name__ == "__main__":

    env = SC2Env(env_id="3m")

    state, obs = env.reset()
    done = False
    cnt = 0

    while not done:
        action_masks = {agent: _obs["action_mask"] for agent, _obs in obs.items()}
        actions = {}

        for agent_id, action_mask in action_masks.items():
            ava_actions = np.where(action_mask > 0)[0]
            if len(ava_actions) == 0:
                actions[agent_id] = env.action_spaces[agent_id].sample()
            else:
                actions[agent_id] = np.random.choice(ava_actions)

        _, obs, rew, done, info = env.step(actions)
        done = all(done.values())
        cnt += 1
        print(f"* step: {cnt} reward {rew}", done)
