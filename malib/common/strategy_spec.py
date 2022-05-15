from typing import List, Dict, Any, Tuple, Type
from argparse import Namespace
from filelock import FileLock

import numpy as np

from malib.algorithm.common.policy import Policy
from malib.utils.typing import PolicyID


class StrategySpec:
    def __init__(
        self, policy_ids: Tuple[PolicyID], meta_data: Dict[str, Any] = None
    ) -> None:
        self.policy_ids = tuple(policy_ids)
        self.meta_data = meta_data
        self.num_policy = len(policy_ids)

    def gen_policy(self):
        policy_cls: Type[Policy] = self.meta_data["policy_cls"]
        plist = self.meta_data["kwargs"]
        plist = Namespace(**plist)
        return policy_cls(
            registered_name=plist.registered_name,
            observation_space=plist.observation_space,
            action_space=plist.action_space,
            model_config=plist.model_config,
            custom_config=plist.custom_config,
            **plist.others
        )

    def sample(self) -> PolicyID:
        """Sample a policy instance.

        Returns:
            Policy: Policy instance
        """
        prob_list = self.meta_data.get(
            "prob_list", [1 / self.num_policy] * self.num_policy
        )
        idx = np.random.choice(self.num_policy, p=prob_list)
        return self.policy_ids[idx]

    def load_from_checkpoint(self, policy_id: PolicyID):
        raise NotImplementedError
