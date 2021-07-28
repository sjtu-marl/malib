from typing import Any

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.ppo.loss import PPOLoss
from malib.algorithm.ppo.policy import PPO


class PPOTrainer(Trainer):
    def __init__(self, tid):
        super(PPOTrainer, self).__init__(tid)
        self._loss = PPOLoss()
        # self.cnt = 0

    def optimize(self, batch):
        assert isinstance(self._policy, PPO), type(self._policy)

        self.loss.zero_grad()
        loss_stats = self.loss(batch)
        self.loss.step()
        return loss_stats

    def preprocess(self, **kwargs) -> Any:
        pass
