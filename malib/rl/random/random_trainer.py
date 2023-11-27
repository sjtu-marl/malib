from typing import Any, Dict, Type

import torch

from torch import optim

from malib.rl.common.policy import Policy
from malib.rl.pg.trainer import PGTrainer


class RandomTrainer(PGTrainer):
    def __init__(self, training_config: Dict[str, Any], policy_instance: Policy = None):
        super().__init__(training_config, policy_instance)

    def setup(self):
        self.optimizer: Type[optim.Optimizer] = getattr(
            optim, self.training_config["optimizer"]
        )(self.policy.parameters(), lr=self.training_config["lr"])
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None
