import torch
import torch.nn.functional as F

from typing import Dict, Any

from malib.algorithm.common.loss_func import LossFunc
from malib.utils.episode import EpisodeKey
from malib.utils.typing import TrainingMetric
from .policy import DQN


class DQNLoss(LossFunc):
    def setup_optimizers(self, *args, **kwargs):
        self._policy: DQN
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params["optimizer"])
            self.optimizers = optim_cls(
                self.policy.critic.parameters(), lr=self._params["lr"]
            )
        else:
            # update parameters
            self.optimizers.param_groups = []
            self.optimizers.add_param_group({"params": self.policy.critic.parameters()})

    def step(self) -> Any:
        """Step optimizers and update target"""
        _ = [item.backward() for item in self.loss]

        gradients = {
            "critic": {
                name: param.grad.numpy() * self._params["lr"]
                for name, param in self.policy.critic.named_parameters()
            },
        }

        self.optimizers.step()

        return gradients

    def loss_compute(self, batch) -> Dict[str, Any]:
        self.loss = []
        reward = batch[EpisodeKey.REWARD].view(-1, 1)
        act = batch[EpisodeKey.ACTION].view(-1, 1)
        obs = batch[EpisodeKey.CUR_OBS].copy()
        next_obs = batch[EpisodeKey.NEXT_OBS].copy()
        done = batch[EpisodeKey.DONE].view(-1, 1)

        state_action_values = self.policy.critic(obs).gather(1, act)
        next_state_q = self.policy.target_critic(next_obs)

        if batch.get("next_action_mask", None) is not None:
            next_action_mask = batch["next_action_mask"].copy()
            illegal_action_mask = 1 - next_action_mask
            # give very low value to illegal action logits
            illegal_action_logits = -illegal_action_mask * 1e9
            next_state_q += illegal_action_logits

        next_state_action_values = next_state_q.max(1)[0].unsqueeze(1).detach()

        expected_state_values = (
            reward
            + self.policy.custom_config["gamma"] * (1 - done) * next_state_action_values
        )
        loss = F.mse_loss(state_action_values, expected_state_values)

        self.loss.append(loss)

        return {
            TrainingMetric.LOSS: loss.detach().item(),
            "eps": self.policy._calc_eps(),
        }
