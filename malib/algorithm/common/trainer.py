"""
Trainer is a interface for
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from malib.algorithm.common.policy import Policy
from malib.algorithm.common.loss_func import LossFunc


class Trainer(metaclass=ABCMeta):
    def __init__(self, tid: str):
        """Create a trainer instance.

        :param str tid: Specify trainer id.
        """

        self._tid = tid
        self._training_config = {}
        self._policy = None
        self._loss = None

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)

    @abstractmethod
    def optimize(self, batch) -> Dict[str, Any]:
        """ Execution policy optimization then return a dict of statistics """
        pass

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def loss(self) -> LossFunc:
        return self._loss

    def reset(self, policy, training_config):
        """ Reset policy, called before optimize, and read training configuration """

        self._policy = policy
        self._training_config.update(training_config)
        if self._loss is not None:
            self._loss.reset(policy, training_config)
        # else:
        #     raise ValueError("Loss has not been initialized yet.")

    @abstractmethod
    def preprocess(self, batch, **kwargs) -> Any:
        """ Preprocess batch if need """
        pass
