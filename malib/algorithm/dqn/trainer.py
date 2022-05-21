from argparse import Namespace
from typing import Any, Tuple, Dict, Sequence

import copy

import torch
import numpy as np

from torch.nn import functional as F

from malib.utils.typing import AgentID

from malib.algorithm.common import misc
from malib.algorithm.common.trainer import Trainer
from malib.utils.data import to_torch
from malib.utils.schedules import LinearSchedule


class DQNTrainer(Trainer):
    def setup(self):
        exploration_fraction = self._training_config["exploration_fraction"]
        total_timesteps = self._training_config["total_timesteps"]
        exploration_final_eps = self._training_config["exploration_final_eps"]
        self.fixed_eps = self._training_config.get("pretrain_eps")

        self.exploration = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * total_timesteps),
            initial_p=1.0 if self.fixed_eps is None else self.fixed_eps,
            final_p=exploration_final_eps,
        )

        optim_cls = getattr(torch.optim, self.training_config["optimizer"])
        self.target_critic = copy.deepcopy(self.policy.critic)
        self.optimizer: torch.optim.Optimizer = optim_cls(
            self.policy.critic.parameters(), lr=self.training_config["critic_lr"]
        )

    def post_process(self, batch: Dict[str, Any], agent_filter: Sequence[AgentID]) -> Dict[str, np.ndarray]:
        policy = self.policy.to(
            "cuda" if self.policy.custom_config.get("use_cuda", False) else "cpu",
            use_copy=False,
        )
        # set exploration rate for policy
        if not self._training_config.get("param_noise", False):
            update_eps = self.exploration.value(self.counter)
            update_param_noise_threshold = 0.0
        else:
            update_eps = 0.0
        if self.fixed_eps is not None:
            policy.eps = self.fixed_eps
        else:
            policy.eps = update_eps
        return batch

    def train(self, batch: Dict[str, torch.Tensor]):
        batch = {k: to_torch(v) for k, v in batch.items()}
        batch = Namespace(**batch)
        state_action_values, _ = self.policy.critic(batch.observation)
        state_action_values = state_action_values.gather(
            -1, batch.action.long().view((-1, 1))
        ).view(-1)

        next_state_q, _ = self.target_critic(batch.next_observation)
        next_action_mask = batch.get("next_action_mask", None)

        if next_action_mask is not None:
            illegal_action_mask = 1.0 - next_action_mask
            # give very low value to illegal action logits
            illegal_action_logits = -illegal_action_mask * 1e9
            next_state_q += illegal_action_logits

        next_state_action_values = next_state_q.max(-1)[0]
        expected_state_values = (
            batch.reward
            + self._training_config["gamma"]
            * (1.0 - batch.done)
            * next_state_action_values
        )

        self.optimizer.zero_grad()
        loss = F.mse_loss(state_action_values, expected_state_values.detach())
        loss.backward()
        self.optimizer.step()

        misc.soft_update(
            self.target_critic, self.policy.critic, tau=self._training_config["tau"]
        )

        return {
            "loss": loss.detach().item(),
            "mean_target": expected_state_values.mean().cpu().item(),
            "mean_eval": state_action_values.mean().cpu().item(),
            "min_eval": state_action_values.min().cpu().item(),
            "max_eval": state_action_values.max().cpu().item(),
            "max_target": expected_state_values.max().cpu().item(),
            "min_target": expected_state_values.min().cpu().item(),
            "mean_reward": batch.reward.mean().cpu().item(),
            "min_reward": batch.reward.min().cpu().item(),
            "max_reward": batch.reward.max().cpu().item(),
            "eps": self.policy.eps,
        }
