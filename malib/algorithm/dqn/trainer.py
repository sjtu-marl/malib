from typing import Any

from malib.algorithm.common.trainer import Trainer
from .policy import DQN
from .loss import DQNLoss


class DQNTrainer(Trainer):
    def preprocess(self, batch, **kwargs) -> Any:
        return batch

    def __init__(self, tid: str):
        super(DQNTrainer, self).__init__(tid)
        self._loss = DQNLoss()

    def optimize(self, batch, **kwargs):
        assert isinstance(self._policy, DQN), type(self._policy)
        self._policy.soft_update()
        self.loss.zero_grad()
        if hasattr(self, "main_id"):
            loss_states = self.loss(batch[self.main_id])
        else:
            loss_states = self.loss(batch)
        _ = self.loss.step()
        # self._cnt = (self._cnt + 1) % self._update_interval
        self._policy._step += 1
        return loss_states
