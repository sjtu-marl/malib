from typing import Any, Dict
from malib.algorithm.common.trainer import Trainer


class RandomTrainer(Trainer):
    def __init__(self, tid, policy=None):
        super().__init__(tid)
        self._policy = policy

    def optimize(self, batch, other_agent_batches=None):
        return {}

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError

    def optimize(self, batch) -> Dict[str, Any]:
        raise NotImplementedError

    def preprocess(self, batch, **kwargs) -> Any:
        pass
