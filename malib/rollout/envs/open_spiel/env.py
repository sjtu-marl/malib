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

from typing import List, Dict, Any, Tuple

import gym
import numpy as np
import pyspiel

from gym import spaces
from open_spiel.python.rl_environment import (
    Environment as OPEN_SPIEL_ENV,
    StepType,
    TimeStep,
)

from malib.utils.typing import AgentID
from malib.rollout.envs.env import Environment


SCENARIO_CONFIG = {
    "kuhn_poker": {"players": 2},
    "leduc_poker": {"players": 2},
    "goofspiel": {"players": 2, "num_cards": 3},
}


def ObservationSpace(observation_spec: Dict, **kwargs) -> spaces.Dict:
    """Analyzes accepted observation spec and returns a truncated observation space.
    Args:
        observation_spec (Dict): The raw obsevation spec in dict.
    Returns:
        gym.spaces.Dict: The truncated observation space in Dict.
    """

    _spaces = {}

    if len(observation_spec["info_state"]) > 0:
        _spaces["info_state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_spec["info_state"]
        )
    if len(observation_spec["legal_actions"]) > 0:
        _spaces["action_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=observation_spec["legal_actions"]
        )
    if len(observation_spec["serialized_state"]) > 0:
        _spaces["serialize_state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_spec["serialize_state"]
        )

    return spaces.Dict(_spaces)


def ActionSpace(action_spec: Dict) -> gym.Space:
    """Analyzes accepted action spec and returns a truncated action space.
    Args:
        action_spec (types.Dict): The raw action spec in dict.
    Returns:
        gym.Space: The trucated action space.
    """

    if action_spec["dtype"] == float:
        return spaces.Box(
            low=action_spec["min"],
            high=action_spec["max"],
            shape=(action_spec["num_actions"]),
        )
    elif action_spec["dtype"] == int:
        return spaces.Discrete(action_spec["num_actions"])
    else:
        raise TypeError(
            f"Data type for action space is not allowed, expected are `float` or `int`, but {action_spec['dtype']} received."
        )


class OpenSpielEnv(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        env_id = configs["env_id"]
        scenario_configs = configs.get("scenario_configs", None)
        self.game = pyspiel.load_game_as_turn_based(
            env_id, scenario_configs or SCENARIO_CONFIG[env_id]
        )
        self.env = OPEN_SPIEL_ENV(
            game=self.game, observation_type=configs.get("observation_type", None)
        )

        self._possible_agents = [f"player_{i}" for i in range(self.env.num_players)]
        self.player_mapping = {}

        self.player_int_to_str: Dict[int, str] = None
        self.player_str_to_int: Dict[str, int] = None

        num_agents = len(self.possible_agents)
        self._observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    ObservationSpace(self.env.observation_spec())
                    for _ in range(num_agents)
                ],
            )
        )
        self._action_spaces = dict(
            zip(
                self.possible_agents,
                [ActionSpace(self.env.action_spec()) for _ in range(num_agents)],
            )
        )
        self.use_fixed_player = False

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._possible_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def _parse_obs(self, timestep: TimeStep) -> Dict[AgentID, Any]:
        observations = {k: {} for k in self.possible_agents}
        for k, v in timestep.observations.items():
            # accept only info state and legal actions
            if k not in ["legal_actions", "info_state"]:
                continue
            for player_idx, _v in enumerate(v):
                player = self.player_int_to_str[player_idx]
                obs_ph = observations[player]

                if k == "legal_actions":
                    action_space = self.action_spaces[player]
                    act_shape = (
                        (action_space.n,)
                        if isinstance(action_space, spaces.Discrete)
                        else action_space.shape
                    )
                    mask = np.zeros(act_shape, dtype=np.float32)
                    mask[_v] = 1.0
                    obs_ph["action_mask"] = mask
                elif k == "info_state":
                    obs_ph["info_state"] = np.asarray(_v, dtype=np.float32)
        return observations

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        # convert dict to list and filter invalid players, since turn-based accept only 1 player at each time
        actions = [actions[cur_player] for cur_player in self.cur_players]
        timestep: TimeStep = self.env.step(actions)

        observations = timestep.observations
        rewards = timestep.rewards

        # parse observations
        observations = self._parse_obs(timestep)
        # rewards = dict(zip(self.possible_agents, timestep.rewards))
        rewards = {
            self.player_int_to_str[i]: rew for i, rew in enumerate(timestep.rewards)
        }
        done = timestep.step_type == StepType.LAST
        info = {}

        # update cur player
        if not done:
            self.cur_players = [
                self.player_int_to_str[timestep.observations["current_player"]]
            ]
        else:
            # clean current player cursor.
            self.cur_players = None
            win_role = np.argmax(timestep.rewards)
            info = dict.fromkeys(
                [
                    f"win_{self.player_int_to_str[k]}"
                    for k in range(self.env.num_players)
                ],
                0.0,
            )
            info[f"win_{self.player_int_to_str[win_role]}"] = 1.0

        done = dict.fromkeys(self.possible_agents, done)

        return None, observations, rewards, done, info

    def reset(self, max_step: int = None):
        super().reset(max_step=max_step)

        timestep: TimeStep = self.env.reset()
        # update current player
        if self.use_fixed_player:
            player_idx_seq = np.arange(self.env.num_players).tolist()
        else:
            player_idx_seq = np.random.choice(
                self.env.num_players, self.env.num_players, replace=False
            ).tolist()
        self.player_int_to_str = dict(zip(player_idx_seq, self.possible_agents))
        self.player_str_to_int = dict(zip(self.possible_agents, player_idx_seq))
        self.cur_players = [
            self.player_int_to_str[timestep.observations["current_player"]]
        ]
        observations = self._parse_obs(timestep)
        return None, observations

    def seed(self, seed: int = None):
        self._env.seed(seed)

    def close(self):
        pass


if __name__ == "__main__":  # pragma: no cover
    env = OpenSpielEnv(env_id="kuhn_poker")
    print("Observation spaces:", env.observation_spaces)
    _, obs = env.reset()
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
        print(f"* step: {cnt} reward {rew}, current_players: {env.cur_players}")
