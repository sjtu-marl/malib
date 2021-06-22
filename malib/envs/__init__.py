from malib.utils.typing import Dict, AgentID, List, Any


class Environment:
    def __init__(self, *args, **kwargs):
        self.is_sequential = False
        self._extra_returns = []

    @property
    def extra_returns(self):
        return self._extra_returns

    def reset(self, *args, **kwargs):
        pass

    def step(self, actions: Dict[AgentID, Any]):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError
