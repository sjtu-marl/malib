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
            "critic": torch.optim.Adam(
                params=self.critic.parameters(), lr=self.training_config["critic_lr"]
            ),
            "actor": torch.optim.Adam(
                params=self.policy.actor.parameters(),
                lr=self.training_config["actor_lr"],
            ),
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
        next_states = torch.stack(
            [batch[k][Episode.NEXT_STATE] for k in agents], dim=-2
        )
        observations = torch.stack([batch[k][Episode.CUR_OBS] for k in agents], dim=-2)
        next_observations = torch.stack(
            [batch[k][Episode.NEXT_OBS] for k in agents], dim=-2
        )
        use_timestep = len(states.shape) > 3

        # check actions whether integer or vector
        for agent in agents:
            tensor = batch[agent][Episode.ACTION]
            if not torch.is_floating_point(tensor):
                # convert to onehot
                tensor = F.one_hot(
                    tensor, num_classes=self.policy._action_space.n
                ).float()
                batch[agent][Episode.ACTION] = tensor

        actions = torch.stack([batch[k][Episode.ACTION] for k in agents], dim=-2)
        agent_mask = 1 - torch.eye(n_agents, device=states.device)
        # shape trans: (n_agents, n_agents) -> (n_agents^2, 1) -> (n_agents^2, n_action)
        # -> (n_agents, n_action * n_agents)
        agent_mask = (
            agent_mask.view(-1, 1).repeat(1, actions.shape[-1]).view(n_agents, -1)
        )

        if use_timestep:
            batch_size, time_step, _, _ = states.size()
            joint_actions = actions.view(batch_size, time_step, 1, -1)
            joint_actions = joint_actions.repeat(1, 1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0).unsqueeze(0)
        else:
            (
                batch_size,
                _,
                _,
            ) = states.size()
            joint_actions = actions.view(batch_size, 1, -1)
            joint_actions = joint_actions.repeat(1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0)
        joint_actions = joint_actions * agent_mask

        rewards = torch.stack([batch[k].rew.unsqueeze(-1) for k in agents], dim=-2)
        dones = torch.stack([batch[k].done.unsqueeze(-1) for k in agents], dim=-2)

        batch = Batch(
            {
                Episode.CUR_STATE: states,
                Episode.CUR_OBS: observations,
                Episode.ACTION: actions,
                "joint_act": joint_actions,
                Episode.REWARD: rewards.squeeze(),
                Episode.DONE: dones.squeeze(),
                Episode.NEXT_STATE: next_states,
                Episode.NEXT_OBS: next_observations,
            }
        )
        batch.to_torch(device=states.device)

        return batch

    def create_joint_action(self, n_agents, batch_size, time_step, actions):
        agent_mask = 1 - torch.eye(n_agents, device=actions.device)
        agent_mask = (
            agent_mask.view(-1, 1).repeat(1, actions.shape[-1]).view(n_agents, -1)
        )
        if time_step:
            joint_actions = actions.view(batch_size, time_step, 1, -1)
            joint_actions = joint_actions.repeat(1, 1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0).unsqueeze(0)
        else:
            joint_actions = actions.view(batch_size, 1, -1)
            joint_actions = joint_actions.repeat(1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0)
        joint_actions = joint_actions * agent_mask
        return joint_actions

    def train_critic(self, batch: Batch):
        state = batch[Episode.CUR_STATE]
        obs = batch[Episode.CUR_OBS]
        actions = batch[Episode.ACTION]
        joint_actions = batch["joint_act"]
        use_timestep = len(state.shape) > 3

        critic_state = torch.cat([state, obs, joint_actions], dim=-1)
        pred_q_vals = self.critic(critic_state)

        if isinstance(pred_q_vals, Tuple):
            pred_q_vals = pred_q_vals[0]

        # shape: (batch_size, time_step(optional), agent_dim)
        actions_arg = torch.argmax(actions, dim=-1, keepdim=True)

        if use_timestep:
            target_q_vals = self.target_critic(critic_state)
            if isinstance(target_q_vals, Tuple):
                target_q_vals = target_q_vals[0]
            assert len(target_q_vals.shape) >= 3, target_q_vals.shape
            targets_taken = torch.gather(
                target_q_vals, dim=-1, index=actions_arg
            ).squeeze(-1)
            targets, _ = Postprocessor.compute_episodic_return(
                batch,
                targets_taken.cpu().detach().numpy(),
                gamma=self.training_config["gamma"],
                gae_lambda=self.training_config["gae_lambda"],
            )
            targets = torch.as_tensor(targets, device=self.policy.device).to(
                dtype=torch.float32
            )
        else:
            next_state = batch[Episode.NEXT_STATE]
            next_obs = batch[Episode.NEXT_OBS]
            logits, _ = self.policy.actor(next_obs)
            batch_size, n_agents, _ = next_state.size()
            next_joint_actions = self.create_joint_action(
                n_agents, batch_size, 0, logits.detach()
            )
            next_critic_state = torch.cat(
                [next_state, next_obs, next_joint_actions], dim=-1
            )
            next_target_q_vals = self.target_critic(next_critic_state)
            if isinstance(next_target_q_vals, Tuple):
                next_target_q_vals = next_target_q_vals[0]
            next_target_taken = torch.gather(
                next_target_q_vals,
                dim=-1,
                index=torch.argmax(logits, dim=-1, keepdim=True),
            ).squeeze(-1)
            terminal_mask = 1.0 - batch.done.float()
            targets = (
                batch.rew
                + self.training_config["gamma"]
                * terminal_mask
                * next_target_taken.detach()
            )

        preds = torch.gather(pred_q_vals, dim=-1, index=actions_arg).squeeze(-1)
        loss = F.mse_loss(preds.view(-1), targets.view(-1))

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
        logits, _ = self.policy.actor(batch.obs)
        pi = F.softmax(logits, dim=-1)
        pred_q_vals = pred_q_vals.view(-1, self.policy._action_space.n)

        pi = pi.view(-1, self.policy._action_space.n)
        baselines = (pi * pred_q_vals).sum(-1).detach()

        # caculate pg loss
        # TODO(ming): note the action here is a integer
        actions_arg = torch.argmax(batch.act, dim=-1).reshape(-1, 1)
        q_taken = torch.gather(pred_q_vals, dim=-1, index=actions_arg).squeeze(1)
        pi_taken = torch.gather(pi, dim=-1, index=actions_arg).squeeze(1)

        log_pi_taken = torch.log(pi_taken)
        advantage = (q_taken - baselines).detach()

        coma_loss = (advantage * log_pi_taken).sum()

        self.optimizer["actor"].zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), self.training_config["grad_norm"]
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
