from typing import Dict, Any, Sequence, Union, Type

import torch

from abc import ABCMeta, abstractmethod
from functools import reduce

from malib.utils.typing import AgentID
from malib.algorithm.common.policy import Policy
from malib.utils.data import to_torch


class Trainer(metaclass=ABCMeta):
    def __init__(
        self,
        training_config: Dict[str, Any],
        policy_instance: Policy = None,
    ):
        """Initialize a trainer for a type of policies.

        Args:
            learning_mode (str): Learning mode inidication, could be `off_policy` or `on_policy`.
            training_config (Dict[str, Any], optional): The training configuration. Defaults to None.
            policy_instance (Policy, optional): A policy instance, if None, we must reset it. Defaults to None.
        """

        self._policy = policy_instance
        self._training_config = training_config

        self._counter = 0

        if policy_instance is not None:
            self.setup()

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, v):
        self._policy = v

    @property
    def training_config(self) -> Dict[str, Any]:
        return self._training_config

    @property
    def counter(self):
        return self._counter

    def step_counter(self):
        self._counter += 1

    def parameters(self):
        return self.policy.parameters()

    @abstractmethod
    def setup(self):
        """Set up optimizers here."""

    @abstractmethod
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run training, and return info dict.

        Args:
            batch (Union[Dict[AgentID, Batch], Batch]): A dict of batch or batch

        Returns:
            Dict[str, float]: A dict of information
        """

    def post_process(
        self, batch: Dict[str, Any], agent_filter: Sequence[AgentID]
    ) -> Dict[str, Any]:
        """Batch post processing here.

        Args:
            batch (Dict[str, Any]): Sampled batch.

        Raises:
            NotImplementedError: Not implemented.

        Returns:
            Dict[str, Any]: A processed batch dict.
        """

        return batch

    def __call__(
        self,
        buffer: Dict[str, Any],
        agent_filter: Sequence[AgentID] = None,
    ) -> Dict[str, Any]:
        """Implement the training Logic here, and return the computed loss results.

        Args:
            buffer (Dict[str, Any]): The give data buffer
            agent_filter (Sequence[AgentID], Optional): Determine which agents are governed by \
                this trainer. In single agent mode, there will be only one agents be \
                    transferred. Activated only when `sampler` is not None.

        Returns:
            Dict: A dict of training feedback. Could be agent to dict or string to any scalar/vector datas.
        """

        buffer = self.post_process(buffer, agent_filter)
        # TODO(ming): check whether multiagent buffer, then check torch data
        buffer = {
            k: v
            if isinstance(v, torch.Tensor)
            else to_torch(v, device=self.policy.device)
            for k, v in buffer.items()
        }
        feedback = self.train(buffer)
        self._counter += 1
        return feedback

    def reset(self, policy_instance=None, configs=None, learning_mode: str = None):
        """Reset current trainer, with given policy instance, training configuration or learning mode.

        Note:
            Becareful to reset the learning mode, since it will change the sample behavior. Specifically, \
                the `on_policy` mode will sample datas sequntially, which will return a `torch.DataLoader` \
                    to the method `self.train`. For the `off_policy` case, the sampler will sample data \
                        randomly, which will return a `dict` to 

        Args:
            policy_instance (Policy, optional): A policy instance. Defaults to None.
            configs (Dict[str, Any], optional): A training configuration used to update existing one. Defaults to None.
            learning_mode (str, optional): Learning mode, could be `off_policy` or `on_policy`. Defaults to None.
        """

        self._counter = 0
        if policy_instance is not self._policy:
            self._policy = policy_instance or self._policy
            self.setup()

        if configs is not None:
            self.training_config.update(configs)


TrainerType = Type[Trainer]
