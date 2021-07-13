import torch

from typing import Dict, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode


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

    def __call__(self, batch) -> Dict[str, Any]:
        self.loss = []

        FloatTensor = (
            torch.cuda.FloatTensor
            if self.policy.custom_config["use_cuda"]
            else torch.FloatTensor
        )
        cast_to_tensor = lambda x: FloatTensor(x.copy())

        # total loss = policy_gradient_loss - entropy * entropy_coefficient + value_coefficient * value_loss
        rewards = cast_to_tensor(batch[Episode.REWARD]).view(-1, 1)
        actions = cast_to_tensor(batch[Episode.ACTION])
        cur_obs = cast_to_tensor(batch[Episode.CUR_OBS])
        next_obs = cast_to_tensor(batch[Episode.NEXT_OBS])
        dones = cast_to_tensor(batch[Episode.DONE]).view(-1, 1)
        cliprange = self._params["grad_norm_clipping"]
        alpha = self._params["sac_alpha"]
        gamma = self.policy.custom_config["gamma"]

        # critic update
        vf_in = torch.cat([cur_obs, actions], dim=-1)
        pred_q_1 = self.policy.critic_1(vf_in)
        pred_q_2 = self.policy.critic_2(vf_in)
        next_action_logits = self.policy.compute_actions_by_target_actor(next_obs)
        next_action_dist = Normal(*next_action_logits)
        next_actions = next_action_dist.sample()
        next_action_log_prob = next_action_dist.log_prob(next_actions)
        target_vf_in = torch.cat(
            [next_obs, next_actions],
            dim=-1,
        )
        next_q = (
            torch.min(
                self.policy.target_critic_1(target_vf_in),
                self.policy.target_critic_2(target_vf_in),
            )
            - alpha * next_action_log_prob
        )
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
        policy_action_dist = Normal(*policy_action_logits)
        policy_actions = policy_action_dist.rsample()
        policy_action_log_prob = policy_action_dist.log_prob(policy_actions)
        vf_in = torch.cat([cur_obs, policy_actions], dim=-1)
        current_q_1 = self.policy.critic_1(vf_in)
        current_q_2 = self.policy.critic_2(vf_in)
        actor_loss = (
            alpha * policy_action_log_prob - torch.min(current_q_1, current_q_2)
        ).mean()
        self.optimizers["actor"].zero_grad()
        actor_loss.backward()
        self.optimizers["actor"].step()

        loss_names = [
            "policy_loss",
            "value_loss_1",
            "value_loss_2",
        ]

        stats_list = [
            actor_loss.detach().numpy(),
            critic_loss_1.detach().numpy(),
            critic_loss_2.detach().numpy(),
        ]

        return dict(zip(loss_names, stats_list))
