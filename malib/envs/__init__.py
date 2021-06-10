from malib.utils.typing import Dict, AgentID, List


class Environment:
    def __init__(self, *args, **kwargs):
        self.is_sequential = False

    @property
    def extra_returns(self):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        pass

    def step(self, actions: Dict[AgentID, List]):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError
