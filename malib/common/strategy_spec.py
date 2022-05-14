from typing import List, Dict, Any, Tuple
from malib.algorithm.common.policy import Policy

from malib.utils.typing import PolicyID


class StrategySpec:
    def __init__(
        self, policy_ids: Tuple[PolicyID], meta_data: Dict[str, Any] = None
    ) -> None:
        self.policy_ids = policy_ids
        self.meta_data = meta_data

    def sample(self) -> Policy:
        """Sample a policy instance.

        Returns:
            Policy: Policy instance
        """
        pass
