import torch
from torch import nn
from torch import autograd


from typing import Dict, Tuple, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import TrainingMetric


class AdvIRLLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._params.update({"disc_lr": 1e-2})
        self._reward = None
        self._use_grad_pen = False

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = None
        # self.bce.to(device)
        # self.bce_targets = self.bce_targets.to(device)

    @property
    def reward(self):
        return self._reward

    def setup_optimizers(self, *args, **kwargs):
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = []
            self.optimizers.append(
                optim_cls(
                    self.reward.discriminator.parameters(), lr=self._params["disc_lr"]
                )
            )
            self.optimizers = {
                "discriminator": optim_cls(
                    self.reward.discriminator.parameters(),
                    lr=self._params["disc_lr"],
                )
            }
        else:
            for p in self.optimizers:
                p.param_groups = []
            self.optimizers["discriminator"].add_param_group(
                {"params": self.reward.discriminator.parameters()}
            )

    def step(self) -> Any:
        """Step optimizers and update target"""

        # do loss backward and target update
        _ = [item.backward() for item in self.loss]

        self.push_gradients(
            {
                "discriminator": {
                    name: -self._params["disc_lr"] * param.grad.numpy()
                    for name, param in self.reward.discriminator.named_parameters()
                },
            }
        )

        _ = [p.step() for p in self.optimizers.values()]

    def __call__(self, agent_batch, expert_batch) -> Dict[str, Any]:

        # allocate static memory for bce_targets
        if self.bce_targets is None:
            self.bce_targets = torch.cat(
                [
                    torch.ones(self._params["batch_size"], 1),
                    torch.zeros(self._params["batch_size"], 1),
                ],
                dim=0,
            )

        FloatTensor = (
            torch.cuda.FloatTensor
            if self.reward.custom_config["use_cuda"]
            else torch.FloatTensor
        )
        cast_to_tensor = lambda x: FloatTensor(x.copy())

        expert_cur_obs = cast_to_tensor(expert_batch[Episode.CUR_OBS])
        expert_actions = cast_to_tensor(expert_batch[Episode.ACTION])
        expert_disc_input = torch.cat([expert_cur_obs, expert_actions], dim=-1)

        agent_cur_obs = cast_to_tensor(agent_batch[Episode.CUR_OBS])
        agent_actions = cast_to_tensor(agent_batch[Episode.ACTION])
        agent_disc_input = torch.cat([agent_cur_obs, agent_actions], dim=-1)

        disc_input = torch.cat([expert_disc_input, agent_disc_input], dim=0)

        disc_logits = self.reward.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self._use_grad_pen:
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
            disc_grad_pen_loss = torch.zeros_like(disc_ce_loss)

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()

        loss_names = [
            "disc_ce_loss",
            "disc_grad_pen_loss",
            "disc_total_loss",
            "disc_accuracy",
        ]

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
