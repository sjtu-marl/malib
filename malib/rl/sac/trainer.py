from typing import Any

from malib.rl.common.trainer import Trainer
from .loss import SACLoss


class SACTrainer(Trainer):
    def __init__(self, tid):
        super(SACTrainer, self).__init__(tid)
        self._loss = SACLoss()

    def preprocess(self, batch, **kwargs) -> Any:
        return batch

    def optimize(self, batch):
        self.loss.zero_grad()
        if hasattr(self, "main_id"):
            batch = batch[self.main_id]
        loss_stats = self.loss(batch)
        self.loss.step()
        return loss_stats
