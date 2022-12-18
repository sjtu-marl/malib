# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

from abc import ABC, abstractmethod
from types import LambdaType
from typing import Callable, Union, Dict, Any
from copy import deepcopy


DEFAULT_STOPPING_CONDITIONS = {}


class Scenario(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Any],
        agent_mapping_func: LambdaType,
        training_config: Dict[str, Any],
        rollout_config: Dict[str, Any],
        stopping_conditions: Dict[str, Any],
        dataset_config: Dict[str, Any],
        parameter_server_config: Dict[str, Any],
    ):
        self.name = name
        self.log_dir = log_dir
        self.env_desc = env_desc
        self.algorithms = algorithms
        self.agent_mapping_func = agent_mapping_func
        self.training_config = training_config
        self.rollout_config = rollout_config
        self.stopping_conditions = stopping_conditions or DEFAULT_STOPPING_CONDITIONS
        self.dataset_config = dataset_config or {"table_capacity": 1000}
        self.parameter_server_config = parameter_server_config or {}
        self.parameter_server = None
        self.offline_dataset_server = None

    def copy(self):
        return deepcopy(self)

    def with_updates(self, **kwargs) -> "Scenario":
        new_copy = self.copy()
        for k, v in kwargs.items():
            if not hasattr(new_copy, k):
                raise KeyError(f"{k} is not an attribute of {new_copy.__class__}")
            setattr(new_copy, k, v)
        return new_copy
