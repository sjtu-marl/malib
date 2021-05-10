from abc import ABCMeta, abstractmethod
from typing import Dict, Any

import torch


class LossFunc(metaclass=ABCMeta):
    """Define loss function and optimizers

    Flowchart:
    1. create a loss func instance with: loss = LossFunc(policy, **kwargs)
    2. setup optimizers: loss.setup_optimizers(**kwargs)
    3. zero grads: loss.zero_grads()
    4. calculate loss and got returned statistics: statistics = loss(batch)
    5. do optimization (step): loss.step()
    **NOTE**: if you wanna calculate policy for another policy, do reset: loss.reset(policy)
    """

    def __init__(self):
        self._policy = None
        self.optimizers = []
        self.loss = []
        self._params = {}

    @property
    def optim_cls(self) -> type:
        """Return default optimizer class. If not specify in params, return Adam as default."""

        return getattr(torch.optim, self._params.get("optimizer", "Adam"))

    @property
    def policy(self):
        return self._policy

    @abstractmethod
    def setup_optimizers(self, *args, **kwargs):
        """ Set optimizers and loss function """

        # self.optimizers.append(...)
        # self.loss.append(...)
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """ Compute loss function here, but not optimize """
        pass

    def step(self) -> Any:
        _ = [item.backward() for item in self.loss]
        _ = [p.step() for p in self.optimizers]

    def zero_grad(self):
        _ = [p.zero_grad() for p in self.optimizers]

    def reset(self, policy, configs):
        self._policy = policy
        # reset optimizers
        self.optimizers = []
        self.loss = []
        self._params.update(configs)
        self.setup_optimizers()
