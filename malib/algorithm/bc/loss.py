from typing import Dict, Any

import torch

from malib.algorithm.common.loss_func import LossFunc
from malib.utils.typing import TrainingMetric


class BCLoss(LossFunc):
    def build_loss_func(self, *args, **kwargs):
        self.optimizers.append(
            torch.optim.Adam(self.policy.actor().parameters(), lr=self.policy.lr)
        )
        # self.loss.append(torch.nn.CrossEntropyLoss() if self.policy.discrete_action else torch.nn.MSELoss())

    def __call__(self, eval_actions, expert_actions) -> Dict[str, Any]:
        loss = self.loss[0](eval_actions, expert_actions)
        loss.backward()
        return {TrainingMetric.LOSS: loss.detach().numpy()}
