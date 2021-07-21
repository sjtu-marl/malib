import torch
import torch.optim as optim
from torch import nn
from torch import autograd


from typing import Dict, Tuple, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import TrainingMetric


class DiscriminatorLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._params.update({"reward_lr": 1e-2})
        self._reward = None

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self._params["batch_size"], 1),
                torch.zeros(self._params["batch_size"], 1),
            ],
            dim=0,
        )
        # self.bce.to(device)
        # self.bce_targets = self.bce_targets.to(device)

    def setup_optimizers(self, *args, **kwargs):
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = []
            self.optimizers.append(
                optim_cls(self.reward.parameters(), lr=self._params["lr"])
            )
        else:
            for p in self.optimizers:
                p.param_groups = []
            self.optimizers[0].add_param_group({"params": self.reward.parameters()})

    def step(self) -> Any:
        """ Step optimizers and update target """

        # do loss backward and target update
        _ = [item.backward() for item in self.loss]

        self.push_gradients(
            {
                "reward_func": {
                    name: -self._params["reward_lr"] * param.grad.numpy()
                    for name, param in self.reward.named_parameters()
                },
            }
        )

        _ = [p.step() for p in self.optimizers]

    def __call__(self, expert_batch, agent_batch) -> Dict[str, Any]:
        # empty loss
        self.loss = []

        disc_input = torch.cat([expert_batch, agent_batch], dim=0)

        disc_logits = self.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = torch.randn(expert_batch.size(0), 1)
            # eps.to(device)

            interp_obs = eps * expert_batch + (1 - eps) * agent_batch
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()

        loss_names = [
            "disc_ce_loss",
            "disc_grad_pen_loss",
            "disc_total_loss",
            "disc_accuracy",
        ]

        self.loss.append(disc_ce_loss, disc_grad_pen_loss, disc_total_loss)

        stats_list = [
            disc_ce_loss.detach().numpy(),
            disc_grad_pen_loss.detach().numpy(),
            disc_total_loss.detach().numpy(),
            accuracy.detach().numpy(),
        ]

        return {
            TrainingMetric.LOSS: disc_total_loss.detach().numpy(),
            **dict(zip(loss_names, stats_list)),
        }

    def reset(self, reward, configs):
        # reset optimizers
        # self.optimizers = []
        self.loss = []
        self._params.update(configs)
        if self._reward is not reward:
            self._reward = reward
            self.setup_optimizers()
