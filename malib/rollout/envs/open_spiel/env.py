from typing import List, Dict, Any, Tuple

import gym
import numpy as np
import pyspiel

from gym import spaces
from open_spiel.python.rl_environment import (
    Environment as OPEN_SPIEL_ENV,
    TimeStep,
    SIMULTANEOUS_PLAYER_ID,
)

from malib.utils.typing import AgentID
from malib.rollout.envs.env import Environment


def list_to_dict(datas: List[Any], keys: List[str]):
    return dict(zip(keys, datas))


def dict_to_list(datas: Dict[str, Any], keys: List[str]):
    res = []
    for key in keys:
        res.append(datas[key])
    return res


class OpenSpielEnv(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs["scenario_configs"]
        open_spiel_env_config = scenario_configs["open_spiel_env_config"]
        fixed_player = scenario_configs.get("fixed_player", False)

        self._env = OPEN_SPIEL_ENV(
            game=pyspiel.load_game_as_turn_based(env_id, open_spiel_env_config)
        )
        self._possible_agents = [f"player_{i}" for i in range(self._env.num_players)]

        obs_shape = self._env.observation_spec()["info_state"]
        num_actions = self._env.action_spec()["num_actions"]

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0.0, high=1.0, shape=obs_shape, dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(num_actions,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(num_actions) for agent in self.possible_agents
        }
        self._env_id = env_id
        self._open_spiel_env_config = open_spiel_env_config
        self._trainable_agents = {}
        self._cur_agent_id = None

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._possible_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def observe(self, timestep: TimeStep):
        state = {}
        for i, agent in enumerate(self.possible_agents):
            action_mask = np.zeros(
                self.action_spaces[agent].n, dtype=self.action_spaces[agent].dtype
            )
            action_mask[timestep.observations["legal_actions"][i]] = 1
            state[agent] = {
                "observation": timestep.observations["info_state"][i],
                "action_mask": action_mask,
            }
        return state

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        # convert action dict to action list
        possible_agents = self.possible_agents
        if self._cur_agent_id == SIMULTANEOUS_PLAYER_ID:
            action_list = dict_to_list(actions, possible_agents)
        else:
            cur_agent = possible_agents[self._cur_agent_id]
            action_list = [actions[cur_agent]]
        cur_timestep = self._env.step(action_list)
        rewards = list_to_dict(cur_timestep.rewards, possible_agents)
        dones = dict.fromkeys(possible_agents, cur_timestep.last())

        next_obs = self.observe(cur_timestep)
        rewards = list_to_dict(cur_timestep.rewards, possible_agents)

        # update cur_player
        self._cur_agent_id = cur_timestep.current_player()
        return next_obs, rewards, dones, {agent: {} for agent in possible_agents}

    def reset(self, max_step: int = None, custom_reset_config: Dict[str, Any] = None):
        super().reset(max_step=max_step, custom_reset_config=custom_reset_config)

        cur_timestep = self._env.reset()
        observations = self.observe(cur_timestep)
        self._cur_agent_id = cur_timestep.current_player()
        return (observations,)

    def seed(self, seed: int = None):
        self._env.seed(seed)

    def close(self):
        pass


if __name__ == "__main__":
    SCENARIO_CONFIG = {
        "kuhn_poker": {"players": 2},
        "leduc_poker": {"players": 2},
        "goofspiel": {"players": 2, "num_cards": 3},
    }
    env = OpenSpielEnv(
        env_id="leduc_poker",
        scenario_configs={
            "fixed_player": False,
            "open_spiel_env_config": {"players": 2},
        },
    )

    observations = env.reset()[0]
    action_spaces = env.action_spaces
    done = False
    cnt = 0
    while not done:
        actions = {}
        action_masks = {k: v["action_mask"] for k, v in observations.items()}
        for k, action_mask in action_masks.items():
            candidates = [idx for idx, v in enumerate(action_mask) if v > 0]
            if len(candidates):
                actions[k] = np.random.choice(candidates)
            else:
                actions[k] = action_spaces[k].sample()
        rets = env.step(actions)
        observations, action_masks, rewards, dones, infos = rets
        done = all(dones.values())
        cnt += 1
    print("done", cnt)
