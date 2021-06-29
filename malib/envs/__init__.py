from malib.utils.typing import Dict, AgentID, List, Any, Union, Tuple


class Environment:
    def __init__(self, *args, **kwargs):
        self.is_sequential = False
        self._extra_returns = []
        self._trainable_agents = None

    @property
    def trainable_agents(self) -> Union[Tuple, None]:
        """Return trainble agents, if registered return a tuple, otherwise None"""
        return self._trainable_agents

    @property
    def extra_returns(self):
        return self._extra_returns

    def reset(self, *args, **kwargs):
        pass

    def step(self, actions: Dict[AgentID, Any]):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

