import torch

from typing import Dict, Tuple, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode
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

    def step(self):
        pass

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
                    set(self.policy.actor.parameters()).union(self.policy.critic.parameters()), lr=self._params["lr"]
                ),
            }
        else:
            self.optimizers["policy"].param_groups = []
            self.optimizers["policy"].add_param_group(
                {"params": set(self.policy.actor.parameters()).union(self.policy.critic.parameters())}
            )

    def __call__(self, batch) -> Dict[str, Any]:

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
        cast_to_tensor = lambda x : FloatTensor(x.copy())
        cast_to_long_tensor = lambda x : LongTensor(x.copy())

        rewards = cast_to_tensor(batch[Episode.REWARD])
        if self.policy._discrete:
            actions = cast_to_long_tensor(batch[Episode.ACTION].reshape(-1))
        else:
            actions = cast_to_tensor(batch[Episode.ACTION])
        cur_obs = cast_to_tensor(batch[Episode.CUR_OBS])
        next_obs = cast_to_tensor(batch[Episode.NEXT_OBS])
        dones = cast_to_tensor(batch[Episode.DONE])
        cliprange = self._params["cliprange"]
        grad_cliprange = self._params["grad_norm_clipping"]
        ent_coef = self._params["entropy_coef"]
        vf_coef = self._params["value_coef"]
        optim_epoch = self._params["ppo_epoch"]
        gamma = self.policy.custom_config["gamma"]

        old_logits = self.policy.actor(cur_obs)
        if isinstance(old_logits, tuple):
            old_neglogpac = -Normal(*old_logits).log_prob(actions).detach()
        else:
            old_neglogpac = -Categorical(logits=old_logits).log_prob(actions).detach()
        adv = self.policy.compute_advantage(batch).detach()
        next_value = self.policy.value_function(next_obs).detach().flatten()
        target_value = rewards + gamma * (1.0 - dones) * next_value

        for _ in range(optim_epoch):
            logits = self.policy.actor(cur_obs)
            if isinstance(logits, tuple):
                distri = Normal(*logits)
            else:
                distri = Categorical(logits=logits)
            neglogpac = -distri.log_prob(actions)
            ratio = torch.exp(old_neglogpac.detach() - neglogpac)
            entropy = distri.entropy().mean()

            pg_loss = -adv * ratio
            pg_loss2 = -adv * torch.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
            pg_loss = torch.mean(torch.maximum(pg_loss, pg_loss2))
            approx_kl = 0.5 * torch.mean(torch.square(neglogpac - old_neglogpac))
            clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), cliprange).float())

            vpred = self.policy.value_function(cur_obs).flatten()
            vf_loss = (vpred - target_value).pow(2).mean()

            # total loss
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

            self.optimizers["policy"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
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
