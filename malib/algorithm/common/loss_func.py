from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Sequence

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
        self.optimizers = None
        self.loss = []
        self._params = {}
        self._gradients = []

    @property
    def stacked_gradients(self):
        """Return stacked gradients"""

        return self._gradients

    def push_gradients(self, grad):
        """Push new gradient to gradients"""

        self._gradients.append(grad)

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

    @abstractmethod
    def step(self) -> Any:
        pass

    def zero_grad(self):
        """Clean stacked gradients and optimizers"""

        self._gradients = []
        if isinstance(self.optimizers, Sequence):
            _ = [p.zero_grad() for p in self.optimizers]
        elif isinstance(self.optimizers, Dict):
            _ = [p.zero_grad() for p in self.optimizers.values()]
        elif isinstance(self.optimizers, torch.optim.Optimizer):
            self.optimizers.zero_grad()
        else:
            raise TypeError(
                f"Unexpected optimizers type: {type(self.optimizers)}, expected are included: Sequence, Dict, and torch.optim.Optimizer"
            )

    def reset(self, policy, configs):
        # reset optimizers
        # self.optimizers = []
        self.loss = []
        self._params.update(configs)
        if self._policy is not policy:
            self._policy = policy
            self.setup_optimizers()
