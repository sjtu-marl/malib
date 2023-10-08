from typing import Dict, Union, Any, List

from enum import IntEnum
from dataclasses import dataclass, field

from malib.utils.typing import AgentID


class TaskType(IntEnum):
    ROLLOUT = 0
    EVALUATION = 1
    OPTIMIZATION = 2


@dataclass
class RolloutTask:

    task_type: int
    active_agents: List[AgentID]
    strategy_specs: Dict[str, Any] = field(default_factory=dict())
    
    @classmethod
    def from_raw(cls, dict_style: Union[Dict[str, Any], "RolloutTask"], **kwargs) -> "RolloutTask":
        if isinstance(dict_style, Dict):
            return cls(**dict_style, **kwargs)
        elif isinstance(dict_style, cls):
            return dict_style
        else:
            raise TypeError(f"Unexpected type: {type(dict_style)}")


@dataclass
class OptimizationTask:

    @classmethod
    def from_raw(cls, dict_style: Union[Dict[str, Any], "OptimizationTask"], **kwargs) -> "OptimizationTask":
        raise NotImplementedError
