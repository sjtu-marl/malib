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

from argparse import Namespace
from typing import Dict, Any, Sequence
from threading import Lock

import traceback
import torch
import numpy as np

from malib.common.strategy_spec import StrategySpec
from malib.remote.interface import RemoteInterface
from malib.utils.logging import Logger


class Table:
    def __init__(self, policy_meta_data: Dict[str, Any]):
        policy_cls = policy_meta_data["policy_cls"]
        optim_config = policy_meta_data.get("optim_config")
        plist = Namespace(**policy_meta_data["kwargs"])
        self.state_dict = None
        if optim_config is not None:
            self.optimizer: torch.optim.Optimizer = getattr(
                torch.optim, optim_config["type"]
            )(self.policy.parameters(), lr=optim_config["lr"])
        else:
            self.optimizer: torch.optim.Optimizer = None
        self.lock = Lock()

    def set_weights(self, state_dict):
        with self.lock:
            self.state_dict = state_dict

    def apply_gradients(self, *gradients):
        raise NotImplementedError
        # with self.lock:
        #     summed_gradients = [
        #         np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        #     ]
        #     self.optimizer.zero_grad()
        #     for g, p in zip(summed_gradients, self.policy.parameters()):
        #         if g is not None:
        #             p.grad = torch.from_numpy(g.copy())
        #     self.optimizer.step()

    def get_weights(self):
        with self.lock:
            return self.state_dict


class ParameterServer(RemoteInterface):
    def __init__(self, **kwargs):
        self.tables: Dict[str, Table] = {}
        self.lock = Lock()

    def start(self):
        """For debug"""
        Logger.info("Parameter server started")

    def apply_gradients(self, table_name: str, gradients: Sequence[Any]):
        self.tables[table_name].apply_gradients(*gradients)

    def get_weights(self, spec_id: str, spec_policy_id: str) -> Dict[str, Any]:
        """Request for weight retrive, return a dict includes keys: `spec_id`, `spec_policy_id` and `weights`.

        Args:
            spec_id (str): Strategy spec id.
            spec_policy_id (str): Related policy id.

        Returns:
            Dict[str, Any]: A dict.
        """

        table_name = f"{spec_id}/{spec_policy_id}"
        weights = self.tables[table_name].get_weights()
        return {
            "spec_id": spec_id,
            "spec_policy_id": spec_policy_id,
            "weights": weights,
        }

    def set_weights(
        self, spec_id: str, spec_policy_id: str, state_dict: Dict[str, Any]
    ):
        table_name = f"{spec_id}/{spec_policy_id}"
        self.tables[table_name].set_weights(state_dict)

    def create_table(self, strategy_spec: StrategySpec) -> str:
        """Create parameter table with given strategy spec. This function will traverse existing policy \
            id in this spec, then generate table for policy ids which have no cooresponding tables.

        Args:
            strategy_spec (StrategySpec): A startegy spec instance.

        Returns:
            str: Table name.
        """

        try:
            with self.lock:
                for policy_id in strategy_spec.policy_ids:
                    table_name = f"{strategy_spec.id}/{policy_id}"
                    if table_name in self.tables:
                        continue
                    meta_data = strategy_spec.get_meta_data().copy()
                    self.tables[table_name] = Table(meta_data)
            return table_name
        except Exception as e:
            raise e
