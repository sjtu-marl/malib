import torch

from typing import Dict, Tuple, Any

from torch.distributions.categorical import Categorical

from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import TrainingMetric


def cal_entropy(logits):
    max_value, _ = torch.max(logits, dim=-1, keepdim=True)
    a0 = logits - max_value
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=-1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)


class PPOLoss(LossFunc):

    def __init__(self):
        super(PPOLoss, self).__init__()
        # default parameters here
        self._params = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-4,
            "cliprange": 0.2,
            "entropy_coef": 1e-3,
            "value_coef": 0.5,
        }

    def setup_optimizers(self, *args, **kwargs):
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
        self.optimizers.append(
            optim_cls(self.policy.actor.parameters(), lr=self._params["actor_lr"])
        )
        self.optimizers.append(
            optim_cls(self.policy.critic.parameters(), lr=self._params["critic_lr"])
        )

    def step(self) -> Any:
        """ Step optimizers and update target """

        # do loss backward and target update
        _ = [item.backward() for item in self.loss]
        # return gradients here
        # gradients = {
        #     "actor": {
        #         name: -self._params["actor_lr"] * param.grad.numpy()
        #         for name, param in self.policy.actor.named_parameters()
        #     },
        #     "critic": {
        #         name: -self._params["critic_lr"] * param.grad.numpy()
        #         for name, param in self.policy.critic.named_parameters()
        #     },
        # }
        _ = [p.step() for p in self.optimizers]
        return None

    def __call__(self, batch) -> Dict[str, Any]:
        # empty loss
        self.loss = []
        # total loss = policy_gradient_loss - entropy * entropy_coefficient + value_coefficient * value_loss
        rewards = torch.from_numpy(batch[Episode.REWARDS].copy())
        actions = torch.from_numpy(batch[Episode.ACTIONS].copy())
        cliprange = self._params["cliprange"]
        ent_coef = self._params["entropy_coef"]
        vf_coef = self._params["value_coef"]

        old_probs = self.policy.target_actor(batch[Episode.CUR_OBS].copy())
        old_neglogpac = -Categorical(probs=old_probs).log_prob(actions)
        old_vpred = self.policy.target_value_function(
            batch[Episode.CUR_OBS].copy()
        ).detach()
        # torch.from_numpy(batch[Episode.STATE_VALUE].copy())

        probs = self.policy.actor(batch[Episode.CUR_OBS].copy())
        distri = Categorical(probs=probs)
        neglogpac = -distri.log_prob(actions)
        ratio = torch.exp(old_neglogpac.detach() - neglogpac)
        entropy = torch.mean(cal_entropy(distri.logits))

        adv = self.policy.compute_advantage(batch).detach()
        pg_loss = -adv * ratio
        pg_loss2 = -adv * torch.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = torch.mean(torch.maximum(pg_loss, pg_loss2))
        approx_kl = 0.5 * torch.mean(torch.square(neglogpac - old_neglogpac))
        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), cliprange).float())

        vpred = self.policy.value_function(batch[Episode.CUR_OBS].copy())
        vpred_clipped = old_vpred + torch.clip(vpred - old_vpred, -cliprange, cliprange)
        next_value = self.policy.target_value_function(batch[Episode.NEXT_OBS].copy())
        td_value = (
            rewards
            + self.policy.custom_config["gamma"]
            * (1.0 - torch.from_numpy(batch[Episode.DONES].copy()).float())
            * next_value
        )
        vf_loss1 = torch.square(vpred - td_value)
        vf_loss2 = torch.square(vpred_clipped - td_value)
        vf_loss = 0.5 * torch.mean(torch.maximum(vf_loss1, vf_loss2))

        # total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]

        self.loss.append(loss)

        stats_list = [
            pg_loss.detach().numpy(),
            vf_loss.detach().numpy(),
            entropy.detach().numpy(),
            approx_kl.detach().numpy(),
            clip_frac.detach().numpy(),
        ]

        return {
            TrainingMetric.LOSS: loss.detach().numpy(),
            **dict(zip(loss_names, stats_list)),
        }
