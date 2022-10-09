from argparse import Namespace
from typing import Type, Dict, Any, Sequence

import torch
import numpy as np

from torch import optim

from malib.rl.common.trainer import Trainer
from malib.utils.data import Postprocessor
from malib.utils.typing import AgentID
from malib.utils.tianshou_batch import Batch


class PGTrainer(Trainer):
    def setup(self):
        self.optimizer: Type[optim.Optimizer] = getattr(
            optim, self.training_config["optimizer"]
        )(self.policy.parameters()["actor"], lr=self.training_config["lr"])
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None

    def post_process(
        self, batch: Batch, agent_filter: Sequence[AgentID]
    ) -> Dict[str, Any]:

        # v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = Postprocessor.compute_episodic_return(
            batch, gamma=self.training_config["gamma"], gae_lambda=1.0
        )

        if self.training_config["reward_norm"] and self.ret_rms is not None:
            batch["returns"] = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch["returns"] = unnormalized_returns
        batch["logits"], _ = self.policy.actor(
            batch.obs, state=batch.get("state", None)
        )
        return batch

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        logits = batch.logits
        dist = self.policy.dist_fn.proba_distribution(logits)
        act = batch.act
        ret = batch.returns
        log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
        loss = -(log_prob * ret).mean()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {"avg_loss": loss.item()}
