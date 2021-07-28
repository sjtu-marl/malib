# -*- coding: utf-8 -*-
from malib.backend.datapool.offline_dataset_server import Episode
import random
from typing import List

import numpy as np
from gym import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv

from open_spiel.python.rl_environment import Environment as OPEN_SPIEL_ENV, TimeStep
from malib.envs import Environment


class PokerEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        super(PokerEnv, self).__init__()
        self._open_spiel_env = OPEN_SPIEL_ENV(game="leduc_poker")

        self.possible_agents = [
            f"player_{i}" for i in range(self._open_spiel_env.num_players)
        ]
        self.agents = self.possible_agents[:]

        obs_shape = self._open_spiel_env.observation_spec()["info_state"]
        num_actions = self._open_spiel_env.action_spec()["num_actions"]
        self.observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    spaces.Dict(
                        {
                            "observation": spaces.Box(
                                low=0.0, high=1.0, shape=obs_shape, dtype=np.int8
                            ),
                            "action_mask": spaces.Box(
                                low=0, high=1, shape=(num_actions,), dtype=np.int8
                            ),
                        }
                    )
                    for _ in range(self.num_agents)
                ],
            )
        )

        self.action_spaces = dict(
            zip(
                self.possible_agents,
                [spaces.Discrete(num_actions) for _ in range(self.num_agents)],
            )
        )

        self._cur_time_step: TimeStep = None
        self._fixed_player = kwargs.get("fixed_player", False)
        self._player_map = None

    def seed(self, seed=None):
        # warning: nothing will be done since I don't know
        #  how to set the random seed in the underlying game.
        self._open_spiel_env = OPEN_SPIEL_ENV(game="leduc_poker")

    def _scale_rewards(self, reward):
        return reward

    def _int_to_name(self, ind):
        ind = self._player_map(ind)
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self._player_map(self.possible_agents.index(name))

    def _convert_to_dict(self, data_list: List):
        agents = [
            self.possible_agents[self._player_map(i)] for i in range(self.num_agents)
        ]
        return dict(zip(agents, data_list))

    def observe(self, agent):
        obs = self._cur_time_step.observations["info_state"][self._name_to_int(agent)]
        observation = np.array(obs, dtype=np.int8)
        legal_moves = self.next_legal_moves
        action_mask = np.zeros(self._open_spiel_env.action_spec()["num_actions"], int)
        action_mask[legal_moves] = 1

        return {"observation": observation, "action_mask": action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        self._cur_time_step = self._open_spiel_env.step([action])
        self._last_obs = self.observe(self.agent_selection)

        if self._cur_time_step.last():
            self.rewards = self._convert_to_dict(
                self._scale_rewards(self._cur_time_step.rewards)
            )
            self.infos[self.agent_selection]["legal_moves"] = []
            self.next_legal_moves = []
            self.dones = self._convert_to_dict([True for _ in range(self.num_agents)])
            next_player = self._int_to_name(1 - self._name_to_int(self.agent_selection))
        else:
            next_player = self._int_to_name(self._cur_time_step.current_player())
            self.next_legal_moves = self._cur_time_step.observations["legal_actions"][
                self._cur_time_step.current_player()
            ]
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self.infos[self.agent_selection]["legal_moves"] = self.next_legal_moves
        self._accumulate_rewards()
        self._dones_step_first()

    def reset(self):
        if self._fixed_player:
            self._player_map = lambda p: p
        else:
            self._player_map = random.choice([lambda p: p, lambda p: 1 - p])

        self._cur_time_step = self._open_spiel_env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._int_to_name(self._cur_time_step.current_player())
        self.rewards = self._convert_to_dict([0.0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0.0 for _ in range(self.num_agents)]
        )
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict(
            [
                {"legal_moves": _lm}
                for _lm in self._cur_time_step.observations["legal_actions"]
            ]
        )
        self.next_legal_moves = list(
            sorted(self.infos[self.agent_selection]["legal_moves"])
        )
        self._last_obs = np.array(
            self._cur_time_step.observations["info_state"][
                self._name_to_int(self.agent_selection)
            ],
            dtype=np.int8,
        )

    def render(self, mode="human"):
        raise NotImplementedError()

    def close(self):
        pass


def env(**kwargs):
    env = PokerEnv(**kwargs["scenario_configs"])
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = Environment.from_sequential_game(env, **kwargs)
    env._extra_returns = [Episode.ACTION_MASK]
    return env
