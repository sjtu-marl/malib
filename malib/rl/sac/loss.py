import torch

from typing import Dict, Any

from torch.distributions import Independent, Normal

from malib.rl.common.loss_func import LossFunc
from malib.utils.episode import Episode


class SACLoss(LossFunc):
    def __init__(self):
        super(SACLoss, self).__init__()

    def reset(self, policy, configs):
        self._params.update(configs)
        if policy is not self.policy:
            self._policy = policy
            self.setup_optimizers()

    def zero_grad(self):
        _ = [p.zero_grad() for p in self.optimizers.values()]

    def step(self):
        self.policy.soft_update(self._params["tau"])

    def setup_optimizers(self, *args, **kwargs):
        # if self.optimizers is None:
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
        self.optimizers = {
            "actor": optim_cls(
                self.policy.actor.parameters(), lr=self._params["actor_lr"]
            ),
            "critic_1": optim_cls(
                self.policy.critic_1.parameters(), lr=self._params["critic_lr"]
            ),
            "critic_2": optim_cls(
                self.policy.critic_2.parameters(), lr=self._params["critic_lr"]
            ),
        }

    def loss_compute(self, batch) -> Dict[str, Any]:
        self.loss = []

        # total loss = policy_gradient_loss - entropy * entropy_coefficient + value_coefficient * value_loss
        rewards = batch[Episode.REWARD].view(-1, 1)
        actions = batch[Episode.ACTION]
        cur_obs = batch[Episode.CUR_OBS]
        next_obs = batch[Episode.NEXT_OBS]
        dones = batch[Episode.DONE].view(-1, 1)
        alpha = self._params["sac_alpha"]
        gamma = self.policy.custom_config["gamma"]
        action_squash = self.policy.action_squash

        # critic update
        vf_in = torch.cat([cur_obs, actions], dim=-1)
        pred_q_1 = self.policy.critic_1(vf_in).view(-1)
        pred_q_2 = self.policy.critic_2(vf_in).view(-1)

        with torch.no_grad():

            next_action_dist = Independent(self._policy._distribution(next_obs), 1)
            next_actions = next_action_dist.sample()

            next_action_log_prob = next_action_dist.log_prob(next_actions).unsqueeze(-1)
            if action_squash:
                next_actions = torch.tanh(next_actions)
                next_action_log_prob = next_action_log_prob - torch.log(
                    1 - next_actions.pow(2) + self.policy._eps
                ).sum(-1, keepdim=True)
            target_vf_in = torch.cat(
                [next_obs, next_actions],
                dim=-1,
            )
            min_target_q = torch.min(
                self.policy.target_critic_1(target_vf_in),
                self.policy.target_critic_2(target_vf_in),
            )
            next_q = min_target_q - alpha * next_action_log_prob
            target_q = rewards + gamma * next_q * (1.0 - dones)
        critic_loss_1 = (pred_q_1 - target_q.view(-1)).pow(2).mean()
        critic_loss_2 = (pred_q_2 - target_q.view(-1)).pow(2).mean()

        self.optimizers["critic_1"].zero_grad()
        critic_loss_1.backward()
        self.optimizers["critic_1"].step()

        self.optimizers["critic_2"].zero_grad()
        critic_loss_2.backward()
        self.optimizers["critic_2"].step()

        # actor update
        policy_action_dist = Independent(self._policy._distribution(cur_obs), 1)
        policy_actions = policy_action_dist.rsample()
        policy_action_log_prob = policy_action_dist.log_prob(policy_actions).unsqueeze(
            -1
        )
        if action_squash:
            policy_actions = torch.tanh(policy_actions)
            policy_action_log_prob = policy_action_log_prob - torch.log(
                1.0 - policy_actions.pow(2) + self.policy._eps
            ).sum(-1, keepdim=True)
        vf_in = torch.cat([cur_obs, policy_actions], dim=-1)
        current_q_1 = self.policy.critic_1(vf_in)
        current_q_2 = self.policy.critic_2(vf_in)
        actor_loss = (
            (alpha * policy_action_log_prob - torch.min(current_q_1, current_q_2))
            .view(-1)
            .mean()
        )

        self.optimizers["actor"].zero_grad()
        self.optimizers["critic_1"].zero_grad()
        self.optimizers["critic_2"].zero_grad()
        actor_loss.backward()
        self.optimizers["actor"].step()

        loss_names = ["policy_loss", "value_loss_1", "value_loss_2", "reward"]

        stats_list = [
            actor_loss.detach().item(),
            critic_loss_1.detach().item(),
            critic_loss_2.detach().item(),
            rewards.mean().item(),
        ]

        return dict(zip(loss_names, stats_list))
