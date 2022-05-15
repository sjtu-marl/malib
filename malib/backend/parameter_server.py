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

from typing import Dict, Any, Sequence
from threading import Lock

import time
import logging

import ray
import torch
import numpy as np

from torch import nn

from malib import settings
from malib.models.torch import make_net
from malib.common.strategy_spec import StrategySpec


logger = logging.getLogger(__name__)


def log(message: str):
    logger.log(settings.LOG_LEVEL, f"(dataset server) {message}")


class Table:
    def __init__(
        self, model_config: Dict[str, Any], optim_config: Dict[str, Any] = None
    ):
        observation_space = model_config["observation_space"]
        action_space = model_config["action_space"]
        net_type = model_config.get("net_type")
        kwargs = model_config.get("custom_config", {})
        self.model: nn.Module = make_net(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            net_type=net_type,
            **kwargs,
        )
        torch.optim.Adam()
        if optim_config is not None:
            self.optimizer: torch.optim.Optimizer = getattr(
                torch.optim, optim_config["type"]
            )(self.model.parameters(), lr=optim_config["lr"])
        else:
            self.optimizer: torch.optim.Optimizer = None
        self.lock = Lock()

    def set_weights(self, state_dict):
        with self.lock:
            self.model.load_state_dict(state_dict)

    def apply_gradients(self, *gradients):
        with self.lock:
            summed_gradients = [
                np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
            ]
            self.optimizer.zero_grad()
            for g, p in zip(summed_gradients, self.model.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g.copy())
            self.optimizer.step()

    def get_weights(self):
        with self.lock:
            return {k: v.cpu() for k, v in self.state_dict().items()}


@ray.remote
class ParameterServer:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.lock = Lock()

    def start(self):
        """For debug"""
        pass

    def apply_gradients(self, table_name: str, gradients: Sequence[Any]):
        self.tables[table_name].apply_gradients(*gradients)

    def get_weights(self, spec_id: str, spec_policy_id: str):
        table_name = f"{spec_id}/{spec_policy_id}"
        return self.tables[table_name].get_weights()

    def set_weights(self, table_name: str, state_dict: Dict[str, Any]):
        self.tables[table_name].set_weights(state_dict)

    def create_table(self, strategy_spec: StrategySpec) -> str:
        with self.lock:
            for policy_id in strategy_spec.policy_ids:
                table_name = f"{strategy_spec.id}/{policy_id}"
                if table_name in self.tables:
                    continue
                meta_data = strategy_spec.get_meta_data().copy()
                self.tables[table_name] = Table(
                    meta_data["model_config"], meta_data.get("optim_config")
                )
        return table_name
