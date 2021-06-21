from typing_extensions import ParamSpec
from typing import Sequence, Union
from dataclasses import dataclass

from malib.gt.tabular.state import State as GameState


@dataclass
class GameSpec:
    name: str
    game_type: str
    num_players: int


class Game:
    def __init__(self, name: str, game_type: str):
        self._players = []
        self._game_spec = GameSpec(name=name, game_type=game_type, num_players=None)

    def initial_states(self) -> Union[GameState, Sequence[GameState]]:
        """Return initialized states.

        :raises: NotImplementedError
        :returns: A sequence of states or a single state.
        """
        raise NotImplementedError

    @property
    def game_spec(self):
        return self._game_spec
