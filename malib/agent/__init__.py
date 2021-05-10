"""
This package includes agent interfaces for training algorithms and policy pool management.

Example:
    >>> from malib.agent.agent_interface import AgentInterface
    >>> class CustomAgent(AgentInterface):
    ...     def optimize(
    ...         self, policy_ids: Dict[AgentID, PolicyID], batch: Dict[AgentID, Any]
    ...     ) -> Dict[AgentID, Any]:
    ...         pass
    ...
    ...     def add_policy_for_agent(
    ...         self, env_agent_id: AgentID, trainable
    ...     ) -> Tuple[PolicyID, Policy]:
    ...         pass
    ...
    >>> # Usage of single instance
    >>> custom_agent = CustomAgent(...)
    >>> custom_agent.add_policy_for_agent(...)
    >>> custom_agent.optimize(...)
"""

from .indepdent_agent import IndependentAgent
from .ctde_agent import CTDEAgent
from .async_agent import AsyncAgent
from .sync_agent import SyncAgent
from .centralized_agent import CentralizedAgent


def get_training_agent(type_name: str):
    return {
        "independent": IndependentAgent,
        "ctde": CTDEAgent,
        "async": AsyncAgent,
        "sync": SyncAgent,
        "centralized": CentralizedAgent,
    }[type_name]
