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

from typing import List, Dict, Any, Tuple, Type
from argparse import Namespace

import numpy as np

from malib.algorithm.common.policy import Policy
from malib.utils.typing import PolicyID


def validate_meta_data(meta_data):
    assert "policy_cls" in meta_data
    assert "kwargs" in meta_data
    assert "experiment_tag" in meta_data


class StrategySpec:
    def __init__(
        self, identifier: str, policy_ids: Tuple[PolicyID], meta_data: Dict[str, Any]
    ) -> None:
        validate_meta_data(meta_data)
        self.id = identifier
        self.policy_ids = tuple(policy_ids)
        self.meta_data = meta_data
        self.num_policy = len(policy_ids)

    def register_policy_id(self, policy_id: PolicyID):
        """Register new policy id, and preset prob as 0.

        Args:
            policy_id (PolicyID): Policy id to register.
        """

        assert policy_id not in self.policy_ids, (policy_id, self.policy_ids)
        self.policy_ids = self.policy_ids + (policy_id,)

        if "prob_list" in self.meta_data:
            self.meta_data["prob_list"].append(0.0)

    def update_prob_list(self, policy_probs: Dict[PolicyID, float]):
        """Update prob list with given policy probs dict.

        Args:
            policy_probs (Dict[PolicyID, float]): A dict that indicates which policy probs should be updated.
        """

        if "prob_list" not in self.meta_data:
            self.meta_data["prob_list"] = [0.0] * len(self.policy_ids)
        for pid, prob in policy_probs.items():
            idx = self.policy_ids.index(pid)
            self.meta_data["prob_list"][idx] = prob
        assert np.isclose(self.meta_data["prob_list"], 1.0), (
            self.meta_data["prob_list"],
            sum(self.meta_data["prob_list"]),
        )

    def get_meta_data(self) -> Dict[str, Any]:
        return self.meta_data

    def gen_policy(self) -> Policy:
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
        """Sample a policy instance. Use uniform sample if there is no presetted prob list in meta data.

        Returns:
            PolicyID: A sampled policy id.
        """

        prob_list = self.meta_data.get(
            "prob_list", [1 / self.num_policy] * self.num_policy
        )
        idx = np.random.choice(self.num_policy, p=prob_list)
        return self.policy_ids[idx]

    def load_from_checkpoint(self, policy_id: PolicyID):
        raise NotImplementedError
