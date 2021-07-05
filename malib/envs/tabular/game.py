from typing import Sequence, Union
from dataclasses import dataclass

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


class Game:
    def __init__(
        self,
        name: str,
        game_type: str,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        **kwargs,
    ):
        """Create a tabular game instance.

        :param str name: The registered game name
        :param str game_type: Game type, should be value in `utils.typing.GameType`
        :param int num_players: The number of players
        :param kwargs: Game configurations
        """

        self._players = []
        self._game_spec = GameSpec(
            name=name,
            game_type=game_type,
            num_players=len(observation_spaces),
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
        )

    def initial_states(self) -> Union[GameState, Sequence[GameState]]:
        """Return initialized states.

        :raises: NotImplementedError
        :returns: A sequence of states or a single state.
        """
        raise NotImplementedError

    @property
    def game_spec(self) -> GameSpec:
        return self._game_spec
