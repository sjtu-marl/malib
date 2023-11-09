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

from typing import Dict, Any, Tuple, Type
from argparse import Namespace
from collections import namedtuple
import numpy as np

from malib.rl.common.policy import Policy
from malib.utils.typing import PolicyID


def validate_meta_data(policy_ids: Tuple[PolicyID], meta_data: Dict[str, Any]):
    """Validate meta data. check whether there is a valid prob list.

    Args:
        policy_ids (Tuple[PolicyID]): A tuple of registered policy ids.
        meta_data (Dict[str, Any]): Meta data.
    """

    assert "policy_cls" in meta_data
    assert "kwargs" in meta_data
    assert "experiment_tag" in meta_data

    if "prob_list" in meta_data:
        assert len(policy_ids) == len(meta_data["prob_list"])
        assert np.isclose(sum(meta_data["prob_list"]), 1.0)


import copy

from gym import spaces


class StrategySpec:
    def __init__(
        self,
        policy_cls: Type,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any] = None,
        identifier: str = None,
        policy_ids: Tuple[PolicyID] = None,
        **kwargs,
    ) -> None:
        """Construct a strategy spec.

        Args:
            identifier (str): Runtime id as identifier.
            policy_ids (Tuple[PolicyID], optional): A tuple of policy id, could be empty. Defaults to None, then sampling will return a None.
            meta_data (Dict[str, Any]): Meta data, for policy construction.
        """

        self.id = identifier or "StrategySpec"
        self.policy_ids = tuple(policy_ids) if policy_ids else ()
        self.meta_data = {
            "policy_cls": policy_cls,
            "init_kwargs": {
                "observation_space": observation_space,
                "action_space": action_space,
                "model_config": model_config,
                **kwargs,
            },
        }

    def __str__(self):
        return f"<StrategySpec: {self.policy_ids}>"

    def __repr__(self) -> str:
        return f"<StrategySpec: {self.policy_ids}>"

    def __len__(self):
        return len(self.policy_ids)

    @property
    def num_policy(self) -> int:
        return len(self.policy_ids)

    def register_policy_id(self, policy_id: PolicyID):
        """Register new policy id, and preset prob as 0.

        Args:
            policy_id (PolicyID): Policy id to register.
        """

        if policy_id in self.policy_ids:
            raise KeyError("repected policy id detected: {}".format(policy_id))
        self.policy_ids = self.policy_ids + (policy_id,)

        if "prob_list" in self.meta_data:
            self.meta_data["prob_list"].append(0.0)

    def update_prob_list(self, policy_probs: Dict[PolicyID, float]):
        """Update prob list with given policy probs dict. Partial assignment is allowed.

        Args:
            policy_probs (Dict[PolicyID, float]): A dict that indicates which policy probs should be updated.
        """

        if "prob_list" not in self.meta_data:
            self.meta_data["prob_list"] = [0.0] * len(self.policy_ids)

        for pid, prob in policy_probs.items():
            idx = self.policy_ids.index(pid)
            self.meta_data["prob_list"][idx] = prob

        if not np.isclose(sum(self.meta_data["prob_list"]), 1.0):
            raise ValueError(
                f"Prob list is not normalized: {self.meta_data['prob_list']}"
            )

    def get_meta_data(self) -> Dict[str, Any]:
        """Return meta data. Keys in meta-data contains

        - policy_cls: policy class type
        - kwargs: a dict of parameters for policy construction
        - experiment_tag: a string for experiment identification
        - optim_config: optional, a dict for optimizer construction

        Returns:
            Dict[str, Any]: A dict of meta data.
        """

        return copy.deepcopy(self.meta_data)

    def gen_policy(self, device=None) -> Policy:
        """Generate a policy instance with the given meta data.

        Returns:
            Policy: A policy instance.
        """

        policy_cls: Type[Policy] = self.meta_data["policy_cls"]
        policy = policy_cls(**self.meta_data["init_kwargs"])
        return policy.to(device)

    def sample(self) -> PolicyID:
        """Sample a policy instance. Use uniform sample if there is no presetted prob list in meta data.

        Returns:
            PolicyID: A sampled policy id.
        """

        if len(self) == 0:
            raise RuntimeError(
                "No policy id registered, it would be feasible for an active policy."
            )

        prob_list = self.meta_data.get(
            "prob_list", [1 / self.num_policy] * self.num_policy
        )
        assert (
            np.isclose(sum(prob_list), 1.0) and len(prob_list) == self.num_policy
        ), f"You cannot specify a prob list whose sum is not close to 1.: {prob_list}. Or an inconsistent length detected: {len(prob_list)} (expcted: {self.num_policy})."

        idx = np.random.choice(self.num_policy, p=prob_list)
        return self.policy_ids[idx]

    def load_from_checkpoint(self, policy_id: PolicyID):
        raise NotImplementedError
