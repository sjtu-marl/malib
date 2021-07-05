from malib.gt.tabular import types as tabularType
from malib.utils.typing import Dict, Tuple, Sequence, Any


class State:
    def __init__(self, actions: Sequence):
        self._action_to_next_state: Dict[tabularType.Action, "State"] = dict()
        self._actions = tuple(actions)

    def legal_actions_mask(self) -> Tuple:
        """Return a tuple of legal action index with mask."""
        raise NotImplementedError

    def information_state_tensor(self) -> Any:
        raise NotImplementedError

    @property
    def actions(self) -> Tuple:
        """Return the tuple of actions, whose length is equal to the action space"""
        return self._actions

    def next(self, action: tabularType.Action) -> "State":
        """Move step and return the next state"""

        assert (
            self._action_to_next_state is not None
        ), "Terminal state has no next states."
        assert (
            action in self.actions
        ), f"Illegal action: {action}, expected should be in {self.actions}"
        return self._action_to_next_state[action]

    def as_terminal(self):
        self._action_to_next_state = None

    @property
    def is_terminal(self):
        return self._action_to_next_state is None
