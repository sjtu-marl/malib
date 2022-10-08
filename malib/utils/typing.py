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

from typing import Dict, Any, Tuple, Sequence
from dataclasses import dataclass

import enum
import gym
import numpy as np


""" Rename and definition of basic data types which are correspond to the inputs (args, kwargs) """
PolicyConfig = Dict[str, Any]
MetaPolicyConfig = Tuple[gym.spaces.Space, gym.spaces.Space, Sequence[PolicyConfig]]
EnvConfig = Dict[str, Any]
RolloutConfig = Dict[str, Any]
ParameterLibConfig = Dict[str, Any]
DatasetConfig = Dict[str, Any]
TrainingConfig = Dict[str, Any]
ModelConfig = Dict[str, Any]
AgentConfig = Dict[str, TrainingConfig]

AgentID = str

PolicyID = str
EnvID = str
EpisodeID = str
DataBlockID = str

DataTransferType = np.ndarray
EnvObservationType = Any
ObservationSpaceType = gym.spaces.Space
ActionSpaceType = gym.spaces.Space


class BehaviorMode(enum.IntEnum):
    """Behavior mode, indicates environment agent behavior"""

    EXPLORATION = 0
    """Trigger exploration mode"""

    EXPLOITATION = 1
    """Trigger exploitation mode"""


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class DataFrame:
    identifier: Any
    data: Any
    meta_data: Dict[str, Any]
