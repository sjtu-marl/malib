from dataclasses import dataclass
from typing import Sequence

import gym

from malib.utils.typing import GameType, AgentID, Dict
from malib.envs.tabular.state import State as GameState


@dataclass
class GameSpec:
    name: str
    game_type: str
    num_players: int
    action_spaces: Dict[AgentID, gym.Space]
    observation_spaces: Dict[AgentID, gym.Space]

    def __post_init__(self):
        """Validation"""
        assert (
            GameType.__dict__.get(self.game_type) is not None
        ), f"Invalid game type: {self.game_type}"


# class Game:
#     def __init__(
#         self,
#         name: str,
#         game_type: str,
#         observation_spaces: Dict[AgentID, gym.Space],
#         action_spaces: Dict[AgentID, gym.Space],
#         **kwargs,
#     ):
#         """Create a tabular game instance.

#         :param str name: The registered game name
#         :param str game_type: Game type, should be value in `utils.typing.GameType`
#         :param int num_players: The number of players
#         :param kwargs: Game configurations
#         """

#         self._players = []
#         self._game_spec = GameSpec(
#             name=name,
#             game_type=game_type,
#             num_players=len(observation_spaces),
#             action_spaces=action_spaces,
#             observation_spaces=observation_spaces,
#         )

#     def initial_state(self) -> GameState:
#         """Return initialized state.

#         :raises: NotImplementedError
#         :returns: A sequence of states or a single state.
#         """
#         raise NotImplementedError

#     @staticmethod
#     def from_game_spec(cls, game_spec: GameSpec):
#         return cls(game_spec.name, game_spec.game_type, game_spec.observation_spaces, game_spec.action_spaces)

#     @property
#     def game_spec(self) -> GameSpec:
#         return self._game_spec

#     @property
#     def states(self):
#         """States is a dict-like mapping from player to states (has `filter` interface)"""

#         raise NotImplementedError


class Game:
    def __init__(
        self,
        name: str,
        env: object,
    ):
        """Create a tabular game instance.

        :param str name: The registered game name
        :param str game_type: Game type, should be value in `utils.typing.GameType`
        :param int num_players: The number of players
        :param kwargs: Game configurations
        """
        assert env.is_sequential, "Tabular game supports only sequential games!"
        self._players = list(env.possible_agents)
        self._game_spec = GameSpec(
            name=name,
            game_type="sequential" if env.is_sequential else "simultaneous",
            num_players=len(env.observation_spaces),
            action_spaces=env.action_spaces,
            observation_spaces=env.observation_spaces,
        )
        # NOTE(ming): this operator is only for sequential games.
        self._env = env.env
        self._states = None

    def initial_state(self) -> GameState:
        """Return initialized state.

        :raises: NotImplementedError
        :returns: A sequence of states or a single state.
        """
        self._env.reset()
        self._states: Dict[AgentID, Sequence[GameState]] = {}
        # pack obsevations to states
        player = next(self._env.agent_iter())
        observation, reward, done, info = self._env.last()
        state = GameState(
            player,
            observation,
            self._game_spec.action_spaces[player],
            observation.get("action_mask"),
            done,
        )
        self._states[player] = []
        self._states[player].append(state)

        return state

    @property
    def game_spec(self) -> GameSpec:
        return self._game_spec

    @property
    def states(self):
        """States is a dict-like mapping from player to states (has `filter` interface)"""

        return self._states
