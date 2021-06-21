from typing import Dict

from malib.gt.tabular import types as tabularType


class State:
    def __init__(self):
        self._action_to_next_state: Dict[tabularType.Action, "State"] = dict()

    def next(self, action: tabularType.Action) -> "State":
        """Move step and return the next state"""

        assert (
            self._action_to_next_state is not None
        ), "Terminal state has no next states."
        return self._action_to_next_state[action]

    def as_terminal(self):
        self._action_to_next_state = None

    @property
    def is_terminal(self):
        return self._action_to_next_state is None
