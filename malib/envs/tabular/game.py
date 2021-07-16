import collections
from dataclasses import dataclass
from typing import Sequence

import gym

from malib.utils.typing import GameType, AgentID, Dict, Tuple
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

Tracer = collections.namedtuple("Tracer", "player, policy")


def _memory_cache(key_fn=lambda *arg: "_".join(arg)):
    """Memoize a single-arg instance method using an on-object cache."""

    def memoizer(method):
        cache_name = "cache_" + method.__name__

        def wrap(self, *arg):
            key = key_fn(*arg)
            cache = vars(self).setdefault(cache_name, {})
            if key not in cache:
                cache[key] = method(self, *arg)
            return cache[key]

        return wrap

    return memoizer


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
        return state

    @property
    def game_spec(self) -> GameSpec:
        return self._game_spec

    @property
    def states(self):
        """States is a dict-like mapping from player to states (has `filter` interface)"""

        return self._states

    def info_sets(
        self, player_id: AgentID, policies: Dict[AgentID, "TabularPolicy"]
    ) -> Dict[str, Tuple[GameState, float]]:
        infosets = collections.defaultdict(list)
        state = self.initial_state()
        self._tracer = Tracer(player_id, policies)
        for s, p in self.iterate_nodes(state):
            infosets[s.information_state_string(player_id)].append((s, p))
        return dict(infosets)

    @_memory_cache()
    def iterate_nodes(self, state: GameState):
        """Iterate state nodes and return a sequence of (state, prob) pairs."""

        if not state.is_terminal():
            if state.current_player() == self._tracer.player:
                yield (state, 1.0)
            for action, p_action in self.transition(state):
                if not state.iterated:
                    # cache environment then rollout
                    self._env.step(action)
                    player = next(self._env.agent_iter())
                    observation, reward, done, info = self._env.last()
                    next_state = GameState(
                        player,
                        observation,
                        self._game_spec.action_spaces[player],
                        observation.get("action_mask"),
                        done,
                    )
                    state.add_transition(action, next_state, reward)
                for next_state, p_state in self.iterate_nodes(state.next(action)):
                    yield (next_state, p_state * p_action)

    @_memory_cache()
    def transition(self, state: GameState):
        """Return a list of (action, prob) pairs from a given parent state.

        Reference: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/best_response.py
        """

        if state.current_player() == self._tracer.player:
            return [(action, 1.0) for action in state.legal_actions_mask()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        else:
            return list(
                self._tracer.policy[state.current_player()]
                .action_probability(state)
                .items()
            )
