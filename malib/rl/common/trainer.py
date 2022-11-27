# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Any, Sequence, Union, Type, List

import torch

from abc import ABCMeta, abstractmethod
from functools import reduce

from malib.utils.typing import AgentID
from malib.rl.common.policy import Policy
from malib.utils.data import to_torch
from malib.utils.tianshou_batch import Batch


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
    def train(self, batch: Batch) -> Dict[str, float]:
        """Run training, and return info dict.

        Args:
            batch (Union[Dict[AgentID, Batch], Batch]): A dict of batch or batch

        Returns:
            Batch: A training batch of data.
        """

    def post_process(self, batch: Batch, agent_filter: Sequence[AgentID]) -> Batch:
        """Batch post processing here.

        Args:
            batch (Batch): Sampled batch.

        Raises:
            NotImplementedError: Not implemented.

        Returns:
            Batch: A batch instance.
        """

        return batch

    def __call__(
        self,
        buffer: Batch,
        agent_filter: Sequence[AgentID] = None,
    ) -> Dict[str, Any]:
        """Implement the training Logic here, and return the computed loss results.

        Args:
            buffer (Batch): The give training batch.
            agent_filter (Sequence[AgentID], Optional): Determine which agents are governed by \
                this trainer. In single agent mode, there will be only one agents be \
                    transferred. Activated only when `sampler` is not None.

        Returns:
            Dict: A dict of training feedback. Could be agent to dict or string to any scalar/vector datas.
        """

        buffer = self.post_process(buffer, agent_filter)
        buffer.to_torch(device=self.policy.device)
        feedback = self.train(buffer)
        if not isinstance(feedback, List):
            feedback = [feedback]
        self.step_counter()
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
