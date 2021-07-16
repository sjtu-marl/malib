import collections
from malib.gt.tabular import types as tabularType
from malib.utils.typing import AgentID, Dict, Tuple, Sequence, Any, List


_DEFAULT_REWARD_FUNC = lambda state, action, next_state: 0.0


class State:
    def __init__(
        self,
        player_id: AgentID,
        information_state_tensor: Any,
        actions: Sequence,
        mask: Sequence,
        game_over: bool,
        reward_func: callable = None,
    ):
        """Create a state instance with given actions and reward function.

        :param Sequence actions: A sequence of actions
        :param callable reward_func: Reward function. If not be specified, will use `_DEFAULT_REWARD_FUNC`
        """

        # XXX(ming): we consider the deterministic state transition,
        #  but the next state could be a sequence of states too.
        self._information_state_tensor = information_state_tensor
        self._action_to_next_state: Dict[tabularType.Action, "State"] = dict()
        self._palyer_id = player_id
        self._actions = tuple(actions)
        self._mask = mask
        self._reward_func = {}
        self._value = 0.0
        self._game_over = game_over
        self._discounted = 1.0
        self._iterated_done = False

    def reward(self, action: "Action") -> float:
        """Compute reward."""

        next_state = self.next(action)

        if self._reward_func.get(action):
            self._reward_func[action] = {}

        return self._reward_func[action].get(next_state, 0.0)

    @property
    def legal_actions_mask(self) -> Tuple:
        """Return a tuple of legal action index with mask."""
        # _apply_mask_to_action_space(self._actions, self._mask)
        raise NotImplementedError

    def information_state_tensor(self) -> Any:
        return self._information_state_tensor

    @property
    def value(self) -> float:
        """Return the state value. Default by 0."""

        return self._value

    @value.setter
    def value(self, value: float):
        """Assign state-value to this state."""

        self._value = value

    @property
    def discounted(self) -> float:
        return self._discounted

    @discounted.setter
    def discounted(self, value: float):
        assert 0.0 <= value <= 1.0, value
        self._discounted = value

    @property
    def game_over(self):
        return self._game_over

    @game_over.setter
    def game_over(self, value: bool):
        self._game_over = value

    @property
    def actions(self) -> Tuple["Action"]:
        """Return the tuple of actions, whose length is equal to the action space"""

        return self._actions

    def is_chance_node(self) -> bool:
        raise NotImplementedError

    def chance_outcomes(self) -> List[Tuple["Action", float]]:
        """Returns the possible change outcomes and their probabilities"""
        raise NotImplementedError

    def current_player(self) -> AgentID:
        """Returns id of the next player to move."""

        return self._player

    def add_transition(self, action, state, reward: float = 0.0):
        assert action in self.legal_actions_mask, (action, self.legal_actions_mask)
        assert (
            self._action_to_next_state.get(action) is None
        ), self._action_to_next_state[action]
        self._action_to_next_state[action] = state
        self._reward_func[action][state] = reward

        self._iterated_done = len(self._action_to_next_state) == len(
            self.legal_actions_mask
        )

    @property
    def iterated_done(self) -> bool:
        return self._iterated_done

    def next(self, action: tabularType.Action) -> "State":
        """Move step and return the next state."""

        # assert (
        #     self._action_to_next_state is not None
        # ), "Terminal state has no next states."
        assert (
            action in self.actions
        ), f"Illegal action: {action}, expected should be in {self.actions}"
        return self._action_to_next_state.get(action)

    def as_terminal(self):
        self._action_to_next_state = None

    @property
    def is_terminal(self) -> bool:
        """Check whether current state is terminal. If there are no child states or game
        has been terminated, return true.
        """

        return self._action_to_next_state is None or self._game_over
