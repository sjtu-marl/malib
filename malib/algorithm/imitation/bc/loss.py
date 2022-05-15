import torch

from typing import Dict, Tuple, Any

from torch.distributions import Categorical, Normal

from malib.algorithm.common.loss_func import LossFunc
from malib.utils.episode import Episode
from malib.utils.typing import TrainingMetric


class BCLoss(LossFunc):
    def __init__(self, mode="mle"):
        super().__init__()
        assert mode in ["mle", "mse"]
        self._params.update({"lr": 1e-2, "mode": mode})

    def setup_optimizers(self, *args, **kwargs):
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = []
            self.optimizers.append(
                optim_cls(self.policy.actor.parameters(), lr=self._params["lr"])
            )
        else:
            for p in self.optimizers:
                p.param_groups = []
            self.optimizers[0].add_param_group(
                {"params": self.policy.actor.parameters()}
            )

    def step(self) -> Any:
        """Step optimizers and update target"""

        # do loss backward and target update
        _ = [item.backward() for item in self.loss]

        self.push_gradients(
            {
                "actor": {
                    name: -self._params["lr"] * param.grad.numpy()
                    for name, param in self.policy.actor.named_parameters()
                },
            }
        )

        _ = [p.step() for p in self.optimizers]

    def loss_compute(self, batch) -> Dict[str, Any]:
        # empty loss
        self.loss = []
        actions = batch[Episode.ACTION]

        probs = self.policy.actor(batch[Episode.CUR_OBS])
        if isinstance(probs, tuple):
            distri = Normal(*probs)
            mu = probs[0]
        else:
            distri = Categorical(probs=probs)

        if self._params["mode"] == "mle":
            neglogpac = -distri.log_prob(actions)
            loss = neglogpac.mean()
        elif self._params["mode"] == "mse":
            if self.policy._discrete:
                loss = torch.nn.CrossEntropyLoss()(prob, actions)
            else:
                loss = torch.square(mu - actions).mean()
        else:
            raise NotImplementedError

        loss_names = [
            "{}_loss".format(self._params["mode"]),
        ]

        self.loss.append(loss)

        stats_list = [
            loss.detach().numpy(),
        ]

        return {
            TrainingMetric.LOSS: loss.detach().numpy(),
            **dict(zip(loss_names, stats_list)),
        }
