from typing import Any

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.ppo.loss import PPOLoss
from malib.algorithm.ppo.policy import PPO


class PPOTrainer(Trainer):
    def __init__(self, tid):
        super(PPOTrainer, self).__init__(tid)
        self._loss = PPOLoss()
        self.cnt = 0

    def optimize(self, batch, other_agent_batches=None):
        assert isinstance(self._policy, PPO), type(self._policy)
        self.cnt = (self.cnt + 1) % self._training_config.get("update_interval", 5)

        if self.cnt == 0:
            self.policy.update_target()

        self.loss.zero_grad()
        loss_stats = self.loss(batch)
        gradients = self.loss.step()
        # loss_stats.update({"gradients": gradients})
        return loss_stats

    def preprocess(self, **kwargs) -> Any:
        pass
