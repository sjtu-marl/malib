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
