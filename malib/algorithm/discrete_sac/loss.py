import torch

from typing import Dict, Any

from torch.distributions import Categorical

from malib.algorithm.common.loss_func import LossFunc
from malib.utils.episode import EpisodeKey


class DiscreteSACLoss(LossFunc):
    def __init__(self):
        super(DiscreteSACLoss, self).__init__()

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
        if self.optimizers is None:
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
            if self.policy.use_auto_alpha:
                self.optimizers["alpha"] = optim_cls(
                    [self.policy._log_alpha], lr=self._params["alpha_lr"]
                )
        else:
            self.optimizers["actor"].param_groups = []
            self.optimizers["actor"].add_param_group(
                {"params": self.policy.actor.parameters()}
            )
            self.optimizers["critic_1"].param_groups = []
            self.optimizers["critic_1"].add_param_group(
                {"params": self.policy.critic_1.parameters()}
            )
            self.optimizers["critic_2"].param_groups = []
            self.optimizers["critic_2"].add_param_group(
                {"params": self.policy.critic_2.parameters()}
            )
            if self.policy.use_auto_alpha:
                self.optimizers["alpha"].param_groups = []
                self.optimizers["alpha"].add_param_group(
                    {"params": [self.policy._log_alpha]}
                )

    def loss_compute(self, batch) -> Dict[str, Any]:
        self.loss = []

        FloatTensor = (
            torch.cuda.FloatTensor
            if self.policy.custom_config["use_cuda"]
            else torch.FloatTensor
        )
        LongTensor = (
            torch.cuda.LongTensor
            if self.policy.custom_config["use_cuda"]
            else torch.LongTensor
        )

        # total loss = policy_gradient_loss - entropy * entropy_coefficient + value_coefficient * value_loss
        rewards = batch[EpisodeKey.REWARD]
        actions = batch[EpisodeKey.ACTION]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        next_obs = batch[EpisodeKey.NEXT_OBS]
        dones = batch[EpisodeKey.DONE]
        cliprange = self._params["grad_norm_clipping"]
        alpha = self.policy._alpha
        gamma = self.policy.custom_config["gamma"]

        # critic update
        pred_q_1 = self.policy.critic_1(cur_obs).gather(1, actions).flatten()
        pred_q_2 = self.policy.critic_2(cur_obs).gather(1, actions).flatten()
        next_action_logits = self.policy.compute_actions_by_target_actor(next_obs)
        next_action_dist = Categorical(logits=next_action_logits)
        next_q = next_action_dist.probs * torch.min(
            self.policy.target_critic_1(next_obs),
            self.policy.target_critic_2(next_obs),
        )
        next_q = next_q.sum(dim=-1) + alpha * next_action_dist.entropy()
        target_q = rewards + gamma * next_q.detach() * (1.0 - dones)
        critic_loss_1 = (pred_q_1 - target_q).pow(2).mean()
        critic_loss_2 = (pred_q_2 - target_q).pow(2).mean()

        self.optimizers["critic_1"].zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic_1.parameters(), cliprange)
        self.optimizers["critic_1"].step()

        self.optimizers["critic_2"].zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic_2.parameters(), cliprange)
        self.optimizers["critic_2"].step()

        # actor update
        policy_action_logits = self.policy.actor(cur_obs)
        policy_action_dist = Categorical(logits=policy_action_logits)
        policy_entropy = policy_action_dist.entropy()
        with torch.no_grad():
            current_q_1 = self.policy.critic_1(cur_obs)
            current_q_2 = self.policy.critic_2(cur_obs)
            current_q = torch.min(current_q_1, current_q_2)
        actor_loss = -(
            alpha * policy_entropy + (policy_action_dist.probs * current_q).sum(dim=-1)
        ).mean()
        self.optimizers["actor"].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), cliprange)
        self.optimizers["actor"].step()

        if self.policy.use_auto_alpha:
            log_prob = -policy_entropy.detach() + self.policy._target_entropy
            alpha_loss = -(self.policy._log_alpha * log_prob).mean()
            self.optimizers["alpha"].zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.policy._log_alpha], cliprange)
            self.optimizers["alpha"].step()
            self.policy._alpha = self.policy._log_alpha.detach().exp()

        loss_names = [
            "policy_loss",
            "value_loss_1",
            "value_loss_2",
            "alpha_loss",
            "alpha",
        ]

        stats_list = [
            actor_loss.detach().numpy(),
            critic_loss_1.detach().numpy(),
            critic_loss_2.detach().numpy(),
            alpha_loss.detach().numpy() if self.policy.use_auto_alpha else 0.0,
            self.policy._alpha.numpy()
            if self.policy.use_auto_alpha
            else self.policy._alpha,
        ]

        return dict(zip(loss_names, stats_list))
