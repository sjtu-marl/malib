from typing import Any

from malib.algorithm.common.trainer import Trainer
from .loss import DDPGLoss
from .policy import DDPG


class DDPGTrainer(Trainer):
    def __init__(self, tid):
        super(DDPGTrainer, self).__init__(tid)
        self._loss = DDPGLoss()

    def preprocess(self, batch, **kwargs) -> Any:
        return batch

    def optimize(self, batch):
        self.loss.zero_grad()
        if hasattr(self, "main_id"):
            batch = batch[self.main_id]
        loss_stats = self.loss(batch)
        self.loss.step()
        return loss_stats
