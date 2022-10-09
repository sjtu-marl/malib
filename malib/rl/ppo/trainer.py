from typing import Dict, Any
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader

from malib.utils.episode import Episode
from malib.rl.common.trainer import Trainer
from malib.rl.common.misc import vtrace, MaskedCategorical
from malib.rl.mappo.utils import PopArt


def grad_norm(model):
    parameters = model.parameters()
    norm_type = 2
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return total_norm


class PPOLoss:
    def __call__(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        old_logits: torch.Tensor,
        old_values: torch.Tensor,
        next_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        worker_action_probs: torch.Tensor,
        action_masks,
        training_config: Dict[str, Any],
    ) -> Any:
        Old_values = old_values.detach()

        if isinstance(logits, tuple):
            dist = Normal(*logits)
            old_dist = Normal(*old_logits)
        else:
            # print("------------ shape check:", logits.shape, action_masks.shape, values.shape, old_logits.shape, old_values.shape, next_values.shape, actions.shape, rewards.shape, dones.shape, worker_action_probs.shape)
            dist = MaskedCategorical(logits, action_masks)
            old_dist = MaskedCategorical(old_logits, action_masks)

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = dist.log_prob(actions)
        Old_logprobs = old_dist.log_prob(actions).detach()
        Worker_logprobs = torch.log(
            worker_action_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        )

        policy_kl_range = training_config["policy_kl_range"]
        cliprange = training_config["cliprange"]
        entropy_coef = training_config["entropy_coef"]
        policy_params = training_config["policy_params"]
        vf_loss_coef = training_config["vf_loss_coef"]
        value_clip = training_config["value_clip"]

        # Getting general advantages estimator
        # Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Advantages = vtrace(
            values,
            rewards,
            next_values,
            dones,
            logprobs,
            Worker_logprobs,
            training_config["gamma"],
            training_config["lam"],
        ).squeeze()
        # print("adv and shae:", Advantages.shape, values.shape)
        assert Advantages.shape == values.shape, (Advantages.shape, values.shape)
        Returns = (Advantages + values).detach()
        Advantages = (
            (Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)
        ).detach()

        # limit = 2.4
        ratios = (logprobs - Old_logprobs).exp()
        ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
        # surr_2 = ratios * Advantages
        # ratios = torch.min(torch.FloatTensor([limit]).to(logprobs.device), (logprobs - Old_logprobs).exp())
        # print("ratio shape: ", logprobs.shape, Old_logprobs.shape)
        # assert logprobs.shape == Old_logprobs.shape, (logprobs.shape, Old_logprobs.shape)
        Kl = 0.5 * torch.mean(torch.square(logprobs - Old_logprobs))
        # kl_divergence(old_dist, dist).float()
        # assert old_dist.probs.shape == dist.probs.shape, (old_dist.shape, dist.shape)
        # torch.mean(torch.greater(torch.abs(ratios - 1.0), cliprange).float())

        # Combining TR-PPO with Rollback (Truly PPO)
        # print("ratio, adv, kl", ratios.mean(), Advantages.mean(), Kl.mean(), ratios.shape, Advantages.shape, Kl.shape)
        adv_loss = ratios.squeeze() * Advantages
        pg_loss = adv_loss.mean()
        # pg_loss = torch.where(
        #     (Kl >= policy_kl_range) & (ratios > 1),
        #     adv_loss - policy_params * Kl,
        #     adv_loss,
        # )
        # # pg_loss = torch.min(ratios * Advantages, clipped_ratios * Advantages)
        # pg_loss = pg_loss.mean()
        # print("---------- pg loss:", pg_loss.detach().cpu().item())

        # Getting entropy from the action probability
        dist_entropy = dist.entropy.mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(
            values - Old_values, -value_clip, value_clip
        )  # Minimize the difference between old value and new value
        # assert values.shape == Old_values.shape == Returns.shape == vpredclipped.shape, (values.shape, Old_values.shape, Returns.shape, vpredclipped.shape)
        vf_losses1 = (Returns.squeeze() - values.squeeze()).pow(
            2
        ) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()
        # critic_loss = vf_losses1.mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (
            (critic_loss * vf_loss_coef)
            - (dist_entropy * entropy_coef)
            - pg_loss
            # + Kl * 0.003
        )
        return (
            loss,
            critic_loss,
            pg_loss,
            dist_entropy,
            Kl,
            adv_loss,
            Returns.mean().item(),
            values.detach().mean().item(),
        )


class CustomDataset(Dataset):
    def __init__(self, batch: Dict[str, np.ndarray], flatten: bool = False) -> None:
        self.states = batch[Episode.CUR_OBS]
        self.actions = batch[Episode.ACTION]
        self.action_masks = batch[Episode.ACTION_MASK]
        self.action_probs = batch[Episode.ACTION_DIST]
        self.rewards = batch[Episode.REWARD]
        self.dones = batch[Episode.DONE]
        self.next_states = batch[Episode.NEXT_OBS]

        # print(
        #     "dataset shape check:",
        #     self.states.shape,
        #     self.actions.shape,
        #     self.action_masks.shape,
        #     self.action_probs.shape,
        #     self.rewards.shape,
        #     self.dones.shape,
        #     self.next_states.shape,
        # )

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.action_masks[idx],
            self.actions[idx],
            self.action_probs[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx],
        )


class PPOTrainer(Trainer):
    def __init__(self, tid):
        super(PPOTrainer, self).__init__(tid)
        self._loss = PPOLoss()
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.target_actor = None
        self.target_critic = None

    def _step(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor,
        worker_action_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_state: torch.Tensor,
    ):
        logits = self.policy.actor(state)
        # print("---------- shape logits, and ams", logits.shape, action_mask.shape)
        assert not logits.isnan().any()
        values = self.policy.critic(state).squeeze()

        old_logits = self.target_actor(state).detach()
        old_values = self.target_critic(state).squeeze()

        next_values = self.policy.critic(next_state).squeeze().detach()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        (
            loss,
            critic_loss,
            pg_loss,
            dist_entropy,
            kl,
            adv_loss,
            avg_return,
            avg_value,
        ) = self.loss(
            logits,
            values,
            old_logits,
            old_values,
            next_values,
            action,
            reward,
            done,
            worker_action_prob,
            action_mask,
            self.training_config,
        )
        loss.backward()

        actor_norm = grad_norm(self.policy.actor)
        critic_norm = grad_norm(self.policy.critic)

        torch.nn.utils.clip_grad_norm_(
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters()),
            self.training_config["grad_norm_clipping"],
        )

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "loss": loss.detach().item(),
            "pg_loss": pg_loss.detach().item(),
            "critic_loss": critic_loss.detach().item(),
            "entropy": dist_entropy.detach().item(),
            "kl": kl.detach().mean().item(),
            "kl_max": kl.detach().max().item(),
            "kl_min": kl.detach().min().item(),
            "adv_loss": adv_loss.detach().mean().item(),
            "actor_norm": actor_norm.detach().item(),
            "critic_norm": critic_norm.detach().item(),
            "avg_return": avg_return,
            "avg_value": avg_value,
        }

    def optimize(self, batch: Dict[str, Any]):
        for k, v in batch.items():
            if self.training_config["batch_mode"] == "episode":
                # print("original k and shape:", k, v.shape)
                if len(v.shape) > 3:
                    v = np.moveaxis(v, 2, 1)
                    batch[k] = v.reshape(-1, *v.shape[3:])
                else:
                    batch[k] = v.reshape(-1, *v.shape[2:])
            if k == Episode.ACTION:
                batch[k] = torch.LongTensor(batch[k].copy()).to(self.policy.device)
            else:
                batch[k] = torch.FloatTensor(batch[k].copy()).to(self.policy.device)
            # print("--- shape:", batch[k].shape)
        batch_size = batch[Episode.CUR_OBS].shape[0]  # num of data point
        mini_batch_size = batch_size // self.training_config["mini_batch"]

        # print(
        #     "mini_batch, size: {}, num: {}".format(
        #         mini_batch_size, self.training_config["mini_batch"]
        #     )
        # )

        dataloader = DataLoader(CustomDataset(batch), mini_batch_size, shuffle=False)
        merged_loss = defaultdict(lambda: 0.0)
        ppo_epoch = self.training_config["ppo_epoch"]
        for _ in range(ppo_epoch):
            for (
                state,
                action_mask,
                action,
                action_prob,
                reward,
                done,
                next_state,
            ) in dataloader:
                loss = self._step(
                    state, action_mask, action, action_prob, reward, done, next_state
                )
                for k, v in loss.items():
                    merged_loss[k] += v / ppo_epoch

        return dict(merged_loss)

    def reset(self, policy, training_config):
        self._training_config.update(training_config)

        if policy is not self._policy:
            print("set policy here for trainer")
            self._policy = policy
            self.target_actor = self._policy.actor.copy()
            self.target_critic = self._policy.critic.copy()

            if self.training_config["use_cuda"]:
                self.target_actor = self.target_actor.cuda()
                self.target_critic = self.target_critic.cuda()

            optim_cls = getattr(
                torch.optim, self.training_config.get("optimizer", "Adam")
            )
            self.actor_optimizer = optim_cls(
                self.policy.actor.parameters(), lr=self.training_config["actor_lr"]
            )
            self.critic_optimizer = optim_cls(
                self.policy.critic.parameters(), lr=self.training_config["critic_lr"]
            )
        # upate target net
        self.target_actor.load_state_dict(self.policy.actor.state_dict())
        self.target_critic.load_state_dict(self.policy.critic.state_dict())
