import torch

from typing import Dict, Tuple, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.utils.episode import Episode
from malib.utils.typing import TrainingMetric


class PPOLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._params.update(
            {
                "cliprange": 0.2,
                "entropy_coef": 1e-3,
                "value_coef": 0.01,
            }
        )

    def reset(self, policy, configs):
        self._params.update(configs)
        if policy is not self.policy:
            self._policy = policy
            self.setup_optimizers()

    def zero_grad(self):
        _ = [p.zero_grad() for p in self.optimizers.values()]

    def setup_optimizers(self, *args, **kwargs):
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "policy": optim_cls(
                    set(self.policy.actor.parameters()).union(
                        self.policy.critic.parameters()
                    ),
                    lr=self._params["lr"],
                ),
            }
        else:
            self.optimizers["policy"].param_groups = []
            self.optimizers["policy"].add_param_group(
                {
                    "params": set(self.policy.actor.parameters()).union(
                        self.policy.critic.parameters()
                    )
                }
            )

    def loss_compute(self, batch) -> Dict[str, Any]:

        rewards = batch[Episode.REWARD]
        if self.policy._discrete_action:
            actions = batch[Episode.ACTION].reshape(-1).long()
        else:
            actions = batch[Episode.ACTION].long()

        cur_obs = batch[Episode.CUR_OBS]
        next_obs = batch[Episode.NEXT_OBS]
        dones = batch[Episode.DONE]
        pi = batch[Episode.ACTION_DIST]

        cliprange = self._params["cliprange"]
        grad_cliprange = self._params["grad_norm_clipping"]
        ent_coef = self._params["entropy_coef"]
        vf_coef = self._params["value_coef"]
        gamma = self.policy.custom_config["gamma"]

        adv = self.policy.compute_advantage(batch).detach()
        next_value = self.policy.value_function(next_obs).detach().flatten()
        target_value = rewards + gamma * (1.0 - dones) * next_value

        logits = self.policy.actor(cur_obs)
        if isinstance(logits, tuple):
            distri = Normal(*logits)
        else:
            distri = Categorical(logits=logits)
        logpi = distri.log_prob(actions)
        old_logpi = torch.log(pi.gather(-1, actions.unsqueeze(-1)).squeeze(-1))
        ratio = torch.exp(logpi - old_logpi.detach())
        entropy = distri.entropy().mean()

        pg_loss = adv * ratio
        pg_loss2 = adv * torch.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = -torch.mean(torch.minimum(pg_loss, pg_loss2))
        approx_kl = 0.5 * torch.mean(torch.square(logpi - old_logpi))
        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), cliprange).float())

        vpred = self.policy.value_function(cur_obs).flatten()
        vf_loss = (vpred - target_value).pow(2).mean()

        # total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        self.optimizers["policy"].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters()),
            grad_cliprange,
        )
        self.optimizers["policy"].step()

        loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]

        stats_list = [
            pg_loss.detach().item(),
            vf_loss.detach().item(),
            entropy.detach().item(),
            approx_kl.detach().item(),
            clip_frac.detach().item(),
        ]

        return {
            TrainingMetric.LOSS: loss.detach().item(),
            **dict(zip(loss_names, stats_list)),
        }
