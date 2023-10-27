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
    strategy_specs: Dict[str, Any] = field(default_factory=dict())
    stopping_conditions: Dict[str, Any] = field(default_factory=dict())
    data_entrypoint_mapping: Dict[str, Any] = field(default_factory=dict())

    @classmethod
    def from_raw(
        cls, dict_style: Union[Dict[str, Any], "RolloutTask"], **kwargs
    ) -> "RolloutTask":
        if isinstance(dict_style, Dict):
            return cls(**dict_style, **kwargs)
        elif isinstance(dict_style, cls):
            return dict_style
        else:
            raise TypeError(f"Unexpected type: {type(dict_style)}")


@dataclass
class OptimizationTask:
    stop_conditions: Dict[str, Any]
    """stopping conditions for optimization task, e.g., max iteration, max time, etc."""

    strategy_specs: Dict[str, Any] = field(default_factory=dict())
    """a dict of strategy specs, which defines the strategy spec for each agent."""

    active_agents: List[AgentID] = field(default_factory=list)
    """a list of active agents, which defines the agents that will be trained in this optimization task. None for all"""

    @classmethod
    def from_raw(
        cls, dict_style: Union[Dict[str, Any], "OptimizationTask"], **kwargs
    ) -> "OptimizationTask":
        """Construct a OptimizationTask object from a dict or a existing OptimizationTask instance.

        Args:
            dict_style (Union[Dict[str, Any], &quot;OptimizationTask&quot;]): A dict or a OptimizationTask instance.

        Raises:
            NotImplementedError: _description_

        Returns:
            OptimizationTask: A OptimizationTask instance.
        """
        raise NotImplementedError
