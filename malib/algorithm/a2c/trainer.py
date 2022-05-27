from argparse import Namespace
from typing import Dict, Any, Sequence

import itertools
import numpy as np
import torch

from torch import optim
from torch import nn
from torch.nn import functional as F

from malib.utils.typing import AgentID
from malib.utils.data import to_torch
from malib.algorithm.common.trainer import Trainer
from malib.utils.data import Postprocessor


class A2CTrainer(Trainer):
    def setup(self):
        parameter_dict = self.policy.parameters()
        # concate parameters
        parameters = set(itertools.chain(*parameter_dict.values()))
        self.optimizer = getattr(optim, self.training_config["optimizer"])(
            parameters, lr=self.training_config["lr"]
        )
        self.parameters = parameters
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None

    def post_process(
        self, batch: Dict[str, Any], agent_filter: Sequence[AgentID]
    ) -> Dict[str, Any]:
        state_value, next_state_value = [], []
        with torch.no_grad():
            for minibatch in batch.split(
                self.training_config.get("max_gae_batchsize", 256),
                shuffle=False,
                merge_last=True,
            ):
                state_value.append(self.policy.critic(minibatch.observation))
                next_state_value.append(self.policy.critic(minibatch.next_observation))
        batch["state_value"] = (
            torch.cat(state_value, dim=0).flatten().cpu().numpy()
        )  # old value
        state_value = batch["state_value"]
        next_state_value = torch.cat(next_state_value, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self.training_config[
            "reward_norm"
        ]:  # unnormalize state_value & next_state_value
            eps = self.training_config["reward_norm"]["config"]["eps"]
            state_value = state_value * np.sqrt(self.ret_rms.var + eps)
            next_state_value = next_state_value * np.sqrt(self.ret_rms.var + eps)

        unnormalized_returns, advantages = Postprocessor.compute_episodic_return(
            batch,
            state_value,
            next_state_value,
            self.training_config["gamma"],
            self.training_config["gae_lambda"],
        )

        if self.training_config["reward_norm"]:
            batch["returns"] = unnormalized_returns / np.sqrt(self.ret_rms.var + eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch["returns"] = unnormalized_returns

        # batch.returns = to_torch_as(batch.returns, batch.state_value)
        batch["advantage"] = advantages  # to_torch_as(advantages, batch.state_value)
        assert (
            batch.advantage.shape == batch.state_value.shape == batch.returns.shape
        ), (batch.advantage.shape, batch.state_value.shape, batch.returns.shape)
        batch["logits"], _ = self.policy.actor(
            batch.observation, state=batch.get("state", None)
        )
        return batch

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = Namespace(
            **{k: to_torch(v, device=self.policy.device) for k, v in batch.items()}
        )
        # calculate loss for actor
        logits = batch.logits
        dist = self.policy.dist_fn.proba_distribution(logits)
        log_prob = dist.log_prob(batch.action)
        log_prob = log_prob.reshape(len(batch.advantage), -1).transpose(0, 1)
        actor_loss = -(log_prob * batch.advantage).mean()
        # calculate loss for critic
        value = self.policy.critic(batch.observation).flatten()
        vf_loss = F.mse_loss(batch.returns, value)
        # calculate regularization and overall loss
        ent_loss = dist.entropy().mean()
        loss = (
            actor_loss
            + self.training_config["value_coef"] * vf_loss
            - self.training_config["entropy_coef"] * ent_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.training_config.get("grad_norm", 0):  # clip large gradient
            nn.utils.clip_grad_norm_(
                self.parameters, max_norm=self.training_config["grad_norm"]
            )
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "actor_loss": actor_loss.item(),
            "vf_loss": vf_loss.item(),
            "ent_loss": ent_loss.item(),
            "total_loss": loss.item(),
        }
