# -*- coding: utf-8 -*-
import torch
import gym
from malib.algorithm.common import misc
from malib.algorithm.common.loss_func import LossFunc
from malib.algorithm.mappo.utils import huber_loss, mse_loss, PopArt
from malib.utils.episode import Episode
from malib.algorithm.common.model import get_model


class MAPPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(MAPPOLoss, self).__init__()
        self.clip_param = 0.2

        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        if self._use_huber_loss:
            self.huber_delta = 10.0
        self._use_value_active_masks = False
        self._use_policy_active_masks = False

        self._use_max_grad_norm = True
        self.max_grad_norm = 10.0

        # self.entropy_coef = 1e-2

        self.use_gae = True

        self.gamma = 0.99
        self.gae_lambda = 0.95

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(),
                    lr=self._params["actor_lr"],
                    eps=self._params["opti_eps"],
                    weight_decay=self._params["weight_decay"],
                ),
                "critic": optim_cls(
                    self.policy.critic.parameters(), lr=self._params["critic_lr"]
                ),
            }
        else:
            self.optimizers["actor"].param_groups = []
            self.optimizers["actor"].add_param_group(
                {"params": self.policy.actor.parameters()}
            )
            self.optimizers["critic"].param_groups = []
            self.optimizers["critic"].add_param_group(
                {"params": self.policy.critic.parameters()}
            )

    def loss_compute(self, sample):
        self._policy.opt_cnt += 1
        # cast = lambda x: torch.FloatTensor(x.copy()).to(self._policy.device)
        (
            share_obs_batch,
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
        ) = (
            sample[Episode.CUR_STATE],
            sample[Episode.CUR_OBS],
            sample[Episode.ACTION].long(),
            sample[Episode.STATE_VALUE],
            sample["return"],
            None,  # cast(sample["active_mask"]),
            sample[Episode.ACTION_DIST],
            sample[Episode.ACTION_MASK],
            sample[f"{Episode.RNN_STATE}_0"],
            sample[f"{Episode.RNN_STATE}_1"],
            sample[Episode.DONE],
        )
        # for k, v in sample.items():
        #     print(f"{k}: {v.shape}")
        #
        if self._policy.custom_config["use_popart"]:
            adv_targ = (return_batch - value_preds_batch).to(return_batch.device)
        else:
            adv_targ = return_batch - value_preds_batch
        adv_targ = (adv_targ - adv_targ.mean()) / (1e-9 + adv_targ.std())

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            active_masks_batch,
        )
        # print(old_action_probs_batch.shape, actions_batch.shape)
        old_action_log_probs_batch = torch.log(
            old_action_probs_batch.gather(-1, actions_batch)
        )
        imp_weights = torch.exp(
            action_log_probs.unsqueeze(-1) - old_action_log_probs_batch
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        self.optimizers["actor"].zero_grad()
        policy_loss = (
            policy_action_loss
            - dist_entropy * self._policy.custom_config["entropy_coef"]
        )
        policy_loss.backward()

        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.actor.parameters(), self.max_grad_norm
            )
        self.optimizers["actor"].step()

        # ============================== Critic optimization ================================
        value_loss = self._calc_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        self.optimizers["critic"].zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self.max_grad_norm
            )
        self.optimizers["critic"].step()

        return dict(
            ratio=imp_weights.detach().mean().cpu().numpy(),
            policy_loss=policy_loss.detach().cpu().numpy(),
            value_loss=value_loss.detach().cpu().numpy(),
            entropy=dist_entropy.detach().cpu().numpy(),
        )

    def _evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        actor_rnn_states_batch,
        critic_rnn_states_batch,
        dones_batch,
        active_masks_batch=None,
    ):
        assert active_masks_batch is None, "Not handle such case"

        logits, _ = self._policy.actor(obs_batch, actor_rnn_states_batch, dones_batch)
        logits -= 1e10 * (1 - available_actions_batch)

        dist = torch.distributions.Categorical(logits=logits)
        # TODO(ziyu): check the shape!!!
        action_log_probs = dist.log_prob(
            actions_batch.view(logits.shape[:-1])
        )  # squeeze the last 1 dimension which is just 1
        dist_entropy = dist.entropy().mean()

        values, _ = self._policy.critic(
            share_obs_batch, critic_rnn_states_batch, dones_batch
        )

        return values, action_log_probs, dist_entropy

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        if self._policy.custom_config["use_popart"]:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = (
                self._policy.value_normalizer(return_batch) - value_pred_clipped
            )
            error_original = self._policy.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def zero_grad(self):
        pass

    def step(self):
        pass
