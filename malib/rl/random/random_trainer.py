from typing import Any, Dict, Sequence, Type

import random
import time
import torch

from torch import optim

from malib.rl.common.policy import Policy
from malib.rl.pg.trainer import PGTrainer
from malib.utils.tianshou_batch import Batch
from malib.utils.typing import AgentID


class RandomTrainer(PGTrainer):
    def __init__(self, training_config: Dict[str, Any], policy_instance: Policy = None):
        super().__init__(training_config, policy_instance)

    def post_process(self, batch: Batch, agent_filter: Sequence[AgentID]) -> Batch:
        return batch

    def train(self, batch: Batch) -> Dict[str, Any]:
        time.sleep(random.random())
        return {"loss": random.random()}

    def setup(self):
        self.optimizer: Type[optim.Optimizer] = getattr(
            optim, self.training_config["optimizer"]
        )(self.policy.parameters(), lr=self.training_config["lr"])
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None
