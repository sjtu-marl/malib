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

from typing import Dict, Sequence, Any, Callable, Tuple
from copy import deepcopy

import torch

from torch.nn import functional as F

from malib.utils.typing import AgentID
from malib.utils.tianshou_batch import Batch
from malib.utils.data import Postprocessor
from malib.utils.episode import Episode
from malib.rl.common import misc
from malib.rl.common.trainer import Trainer
from malib.rl.common.policy import Policy
from .critic import COMADiscreteCritic


class COMATrainer(Trainer):
    def __init__(
        self,
        training_config: Dict[str, Any],
        critic_creator: Callable,
        policy_instance: Policy = None,
    ):
        self.critic_creator = critic_creator
        super().__init__(training_config, policy_instance)

    def setup(self):
        self.critic: torch.nn.Module = self.critic_creator()
        self.target_critic = deepcopy(self.critic)
        self.critic.to(self.policy.device)
        self.target_critic.to(self.policy.device)
        self.optimizer = {
            "critic": torch.optim.Adam(params=self.critic.parameters(), lr=0.001),
            "actor": torch.optim.Adam(params=self.policy.parameters(), lr=0.001),
        }

    def post_process(
        self, batch: Dict[str, Batch], agent_filter: Sequence[AgentID]
    ) -> Batch:
        """Stack batches in agent wise.

        Args:
            batch (Dict[str, Any]): A dict of agent batches.
            agent_filter (Sequence[AgentID]): A list of agent filter.

        Returns:
            Batch: Batch
        """

        agents = list(batch.keys())
        agents.sort()
        n_agents = len(agents)

        # concat by agent-axes: (batch, time_step(optional), inner_dim) -> (batch, time_step(optional), num_agent, inner_dim)
        states = torch.stack([batch[k][Episode.CUR_STATE] for k in agents], dim=-2)
        observations = torch.stack([batch[k][Episode.CUR_OBS] for k in agents], dim=-2)
        use_timestep = len(states.shape) > 3

        actions = torch.stack([batch[k][Episode.ACTION] for k in agents], dim=-2)
        agent_mask = 1 - torch.eye(n_agents, device=states.device)
        # shape trans: (n_agents, n_agents) -> (n_agents^2, 1) -> (n_agents^2, n_action)
        # -> (n_agents, n_action * n_agents)
        agent_mask = (
            agent_mask.view(-1, 1).repeat(1, actions.shape[-1]).view(n_agents, -1)
        )

        if use_timestep:
            batch_size, time_step, _, _ = states.size()
            actions = actions.view(batch_size, time_step, 1, -1)
            actions = actions.repeat(1, 1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0).unsqueeze(0)
        else:
            (
                batch_size,
                _,
                _,
            ) = states.size()
            actions = actions.view(batch_size, 1, -1)
            actions = actions.repeat(1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0)
        actions = actions * agent_mask

        selected_agent_batch = list(batch.values())[0]
        rewards = selected_agent_batch[Episode.REWARD]
        dones = selected_agent_batch[Episode.DONE]

        batch = Batch(
            {
                Episode.CUR_STATE: states,
                Episode.CUR_OBS: observations,
                Episode.ACTION: actions,
                Episode.REWARD: rewards,
                Episode.DONE: dones,
            }
        )

        return batch

    def train_critic(self, batch: Batch, actions: torch.Tensor, rewards: torch.Tensor):
        state = batch[Episode.CUR_STATE]
        obs = batch[Episode.CUR_OBS]
        actions = batch[Episode.ACTION]

        critic_state = torch.cat([state, obs, actions], dim=-1)
        target_q_vals = self.target_critic(critic_state)
        pred_q_vals = self.critic(critic_state)

        if isinstance(target_q_vals, Tuple):
            target_q_vals = target_q_vals[0]

        assert len(target_q_vals.shape) >= 3, target_q_vals.shape

        # shape: (batch_size, time_step(optional), agent_dim)
        targets_taken = torch.gather(target_q_vals, dim=-1, index=actions).squeeze(-1)

        targets = Postprocessor.gae_return(
            targets_taken.cpu().numpy(),
            None,
            batch.rew.cpu().numpy(),
            batch.done.cpu().numpy(),
            self.training_config["gamma"],
            self.training_config["gae_lambda"],
        )
        targets = torch.as_tensor(targets, device=self.policy.device).to(
            dtype=torch.float32
        )
        preds = torch.gather(pred_q_vals, dim=-1, index=actions).squeeze(-1)

        loss = F.mse_loss(preds, targets)

        self.optimizer["critic"].zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.training_config["grad_norm"]
        )
        self.optimizer["critic"].step()

        running_log = {
            "critic_grad_norm": grad_norm.item(),
            "critic_loss": loss.item(),
            "pred_q_mean": preds.mean().item(),
            "target_q_mean": targets.mean().item(),
        }

        return pred_q_vals, running_log

    def train(self, batch: Batch) -> Dict[str, float]:
        pred_q_vals, critic_train_stats = self.train_critic(batch)

        # calculate baseline
        action, pi, logits, hidden_state = self.policy.actor(batch.obs)
        pred_q_vals = pred_q_vals.view(-1, self.policy._action_space.n)

        pi = pi.view(-1, self.policy._action_space.n)
        baselines = (pi * pred_q_vals).sum(-1).detach()

        # caculate pg loss
        # TODO(ming): note the action here is a integer
        q_taken = torch.gather(
            pred_q_vals, dim=-1, index=batch.actions.reshape(-1, 1)
        ).squeeze(1)
        pi_taken = torch.gather(pi, dim=-1, index=batch.actions.reshape(-1, 1)).squeeze(
            1
        )

        log_pi_taken = torch.log(pi_taken)
        advantage = (q_taken - baselines).detach()

        coma_loss = (advantage * log_pi_taken).sum()

        self.optimizer["actor"].zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.training_config["grad_norm"]
        )
        self.optimizer["actor"].step()

        actor_train_stats = {
            "advantage_mean": advantage.mean().item(),
            "coma_loss": coma_loss.item(),
            "actor_grad_norm": grad_norm.item(),
        }

        if self.counter % self.training_config["update_interval"] == 0:
            misc.soft_update(self.target_critic, self.critic, tau=1)

        train_stats = {**actor_train_stats, **critic_train_stats}
        return train_stats
