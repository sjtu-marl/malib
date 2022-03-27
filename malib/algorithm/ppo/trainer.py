from collections import defaultdict
from black import E
import torch
import numpy as np

from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader

from malib.utils.typing import Dict, Any
from malib.utils.episode import EpisodeKey
from malib.algorithm.common.trainer import Trainer
from malib.algorithm.common.misc import vtrace
from malib.algorithm.common.model import get_model


class PPOLoss:
    def __call__(
        self,
        logits,
        values,
        old_logits,
        old_values,
        next_values,
        actions,
        rewards,
        dones,
        worker_action_probs,
        action_masks,
        training_config: Dict[str, Any],
    ) -> Any:
        Old_values = old_values.detach()

        if isinstance(logits, tuple):
            dist = Normal(*logits)
            old_dist = Normal(*old_logits)
        else:
            dist = Categorical(logits=logits)
            old_dist = Categorical(logits=old_logits)

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = dist.log_prob(actions)
        Old_logprobs = old_dist.log_prob(actions).detach()
        Worker_logprobs = torch.log(
            worker_action_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        )

        policy_kl_range = training_config["cliprange"]
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
        )
        # print("adv and shae:", Advantages.shape, values.shape)
        Returns = (Advantages + values).detach()
        Advantages = (
            ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6))
            .detach()
            .squeeze()
        )

        # limit = 2.4
        ratios = (logprobs - Old_logprobs).exp()
        ratios = torch.clamp(ratios, 0.6, 1.4)
        # ratios = torch.min(torch.FloatTensor([limit]).to(logprobs.device), (logprobs - Old_logprobs).exp())
        # print("ratio shape: ", logprobs.shape, Old_logprobs.shape)
        # assert logprobs.shape == Old_logprobs.shape, (logprobs.shape, Old_logprobs.shape)
        Kl = kl_divergence(old_dist, dist).float()
        # assert old_dist.probs.shape == dist.probs.shape, (old_dist.shape, dist.shape)
        # torch.mean(torch.greater(torch.abs(ratios - 1.0), cliprange).float())

        # Combining TR-PPO with Rollback (Truly PPO)
        # print("ratio, adv, kl", ratios.mean(), Advantages.mean(), Kl.mean(), ratios.shape, Advantages.shape, Kl.shape)
        pg_loss = torch.where(
            (Kl >= policy_kl_range) & (ratios > 1),
            ratios * Advantages - policy_params * Kl,
            ratios * Advantages,
        )
        # pg_loss = torch.min(ratios * Advantages, clipped_ratios * Advantages)
        pg_loss = pg_loss.mean()

        # Getting entropy from the action probability
        dist_entropy = dist.entropy().mean()
        # print(ratios.shape, Advantages.shape, Kl.shape)

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(
            values - Old_values, value_clip, value_clip
        )  # Minimize the difference between old value and new value
        # assert values.shape == Old_values.shape == Returns.shape == vpredclipped.shape, (values.shape, Old_values.shape, Returns.shape, vpredclipped.shape)
        # print("values loss:", (values.shape, Old_values.shape, Returns.shape, vpredclipped.shape))
        vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (critic_loss * vf_loss_coef) - (dist_entropy * entropy_coef) - pg_loss
        return loss, critic_loss, pg_loss, dist_entropy


class CustomDataset(Dataset):
    def __init__(self, batch: Dict[str, np.ndarray], flatten: bool = False) -> None:
        self.states = batch[EpisodeKey.CUR_OBS]
        self.actions = batch[EpisodeKey.ACTION]
        self.action_masks = batch[EpisodeKey.ACTION_MASK]
        self.action_probs = batch[EpisodeKey.ACTION_DIST]
        self.rewards = batch[EpisodeKey.REWARD]
        self.dones = batch[EpisodeKey.DONE]
        self.next_states = batch[EpisodeKey.NEXT_OBS]

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
        logits -= 1e10 * (1.0 - action_mask)
        values = self.policy.critic(state)

        # print("---------- device check:", state.device, next(self.target_actor.parameters()).device, next(self.policy.actor.parameters()).device)
        old_logits = self.target_actor(state)
        old_values = self.target_critic(state)

        next_values = self.policy.critic(next_state)

        loss, critic_loss, pg_loss, dist_entropy = self.loss(
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

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "loss": loss.detach().item(),
            "pg_loss": pg_loss.detach().item(),
            "critic_loss": critic_loss.detach().item(),
            "entropy": dist_entropy.detach().item(),
        }

    def optimize(self, batch: Dict[str, Any]):
        for k, v in batch.items():
            v = np.moveaxis(v, 2, 1)
            batch[k] = v.reshape(-1, *v.shape[3:])
            if k == EpisodeKey.ACTION:
                batch[k] = torch.LongTensor(batch[k].copy()).to(self.policy.device)
            else:
                batch[k] = torch.FloatTensor(batch[k].copy()).to(self.policy.device)
            # print("--- shape:", batch[k].shape)
        batch_size = batch[EpisodeKey.CUR_OBS].shape[0]  # num of data point
        mini_batch_size = batch_size // self.training_config["mini_batch"]

        print(
            "mini_batch, size: {}, num: {}".format(
                mini_batch_size, self.training_config["mini_batch"]
            )
        )

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
