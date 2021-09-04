from typing import Dict, Any

import torch

from malib.algorithm.common.loss_func import LossFunc
from malib.algorithm.common import misc
from malib.backend.datapool.offline_dataset_server import Episode


class DDPGLoss(LossFunc):
    def __init__(self):
        super(DDPGLoss, self).__init__()

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
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(), lr=self._params["actor_lr"]
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

    def __call__(self, batch) -> Dict[str, Any]:

        FloatTensor = (
            torch.cuda.FloatTensor
            if self.policy.custom_config["use_cuda"]
            else torch.FloatTensor
        )
        cast_to_tensor = lambda x: FloatTensor(x.copy())

        rewards = cast_to_tensor(batch[Episode.REWARD]).view(-1, 1)
        actions = cast_to_tensor(batch[Episode.ACTION_DIST])
        cur_obs = cast_to_tensor(batch[Episode.CUR_OBS])
        next_obs = cast_to_tensor(batch[Episode.NEXT_OBS])
        dones = cast_to_tensor(batch[Episode.DONE]).view(-1, 1)
        cliprange = self._params["grad_norm_clipping"]
        gamma = self.policy.custom_config["gamma"]

        # ---------------------------------------
        self.optimizers["critic"].zero_grad()
        target_vf_in = torch.cat(
            [next_obs, self.policy.compute_actions_by_target_actor(next_obs)], dim=-1
        )
        next_value = self.policy.target_critic(target_vf_in)
        target_value = rewards + gamma * next_value * (1.0 - dones)

        vf_in = torch.cat([cur_obs, actions], dim=-1)
        actual_value = self.policy.critic(vf_in)

        assert actual_value.shape == target_value.shape, (
            actual_value.shape,
            target_value.shape,
            rewards.shape,
        )
        value_loss = torch.nn.MSELoss()(actual_value, target_value.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), cliprange)
        self.optimizers["critic"].step()
        # --------------------------------------

        # --------------------------------------
        self.optimizers["actor"].zero_grad()
        vf_in = torch.cat([cur_obs, self.policy.compute_actions(cur_obs)], dim=-1)
        # use stop gradient here
        policy_loss = -self.policy.critic(vf_in).mean()  # need add regularization?
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), cliprange)
        self.optimizers["actor"].step()
        # --------------------------------------

        loss_names = ["policy_loss", "value_loss", "target_value_est", "eval_value_est"]

        stats_list = [
            policy_loss.detach().item(),
            value_loss.detach().item(),
            target_value.mean().item(),
            actual_value.mean().item(),
        ]

        return dict(zip(loss_names, stats_list))
