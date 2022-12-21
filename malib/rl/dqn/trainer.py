# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from argparse import Namespace
from typing import Tuple, Sequence

import copy

import torch
import numpy as np

from torch.nn import functional as F

from malib.utils.typing import AgentID

from malib.rl.common import misc
from malib.rl.common.trainer import Trainer
from malib.utils.schedules import LinearSchedule
from malib.utils.tianshou_batch import Batch


class DQNTrainer(Trainer):
    def setup(self):
        exploration_fraction = self._training_config["exploration_fraction"]
        total_timesteps = self._training_config["total_timesteps"]
        exploration_final_eps = self._training_config["exploration_final_eps"]
        self.fixed_eps = self._training_config.get("pretrain_eps")

        self.exploration = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * total_timesteps),
            initial_p=1.0,
            final_p=exploration_final_eps,
        )

        optim_cls = getattr(torch.optim, self.training_config["optimizer"])
        self.target_critic = copy.deepcopy(self.policy.critic)
        self.optimizer: torch.optim.Optimizer = optim_cls(
            self.policy.critic.parameters(), lr=self.training_config["lr"]
        )

    def post_process(self, batch: Batch, agent_filter: Sequence[AgentID]) -> Batch:
        # set exploration rate for policy
        update_eps = self.exploration.value(self.counter)
        self.policy.eps = update_eps
        if self.policy.agent_dimension > 0:
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    inner_shape = v.shape[2:]  # (batch_size, agent_num, inner_shape)
                    batch[k] = v.reshape((-1,) + inner_shape)
        return batch

    def train(self, batch: Batch):
        state_action_values, _ = self.policy.critic(batch.obs.squeeze())
        state_action_values = state_action_values.gather(
            -1, batch.act.long().view((-1, 1))
        ).view(-1)

        next_state_q, _ = self.target_critic(batch.obs_next.squeeze())
        next_action_mask = batch.get("act_mask_next", None)

        if next_action_mask is not None:
            illegal_action_mask = 1.0 - next_action_mask
            # give very low value to illegal action logits
            illegal_action_logits = -illegal_action_mask * 1e9
            next_state_q += illegal_action_logits

        next_state_action_values = next_state_q.max(-1)[0]
        assert batch.rew.shape == batch.done.shape == next_state_action_values.shape, (
            batch.rew.shape,
            batch.done.shape,
            next_state_action_values.shape,
        )

        assert batch.rew.shape == batch.done.shape == next_state_action_values.shape, (
            batch.rew.shape,
            batch.done.shape,
            next_state_action_values.shape,
        )
        expected_state_values = (
            batch.rew.float()
            + self._training_config["gamma"]
            * (1.0 - batch.done.float())
            * next_state_action_values
        )

        assert expected_state_values.shape == state_action_values.shape, (
            expected_state_values.shape,
            state_action_values.shape,
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
            "mean_reward": batch.rew.mean().cpu().item(),
            "min_reward": batch.rew.min().cpu().item(),
            "max_reward": batch.rew.max().cpu().item(),
            "eps": self.policy.eps,
        }
