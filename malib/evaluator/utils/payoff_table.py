import numpy as np

from typing import Sequence, Any, Dict, Tuple
from dataclasses import dataclass

from malib.utils.typing import AgentID, PolicyID


@dataclass
class PayoffTable:
    identify: AgentID
    agents: Sequence[AgentID]
    table: Any = None
    simulation_flag: Any = None

    def __post_init__(self):
        # record policy idx
        self._policy_idx = {agent: {} for agent in self.agents}
        self.table = np.zeros([0] * len(self.agents), dtype=np.float32)
        self.simulation_flag = np.zeros([0] * len(self.agents), dtype=bool)

    def __getitem__(self, key: Dict[str, Sequence[PolicyID]]) -> np.ndarray:
        """Return a sub matrix"""
        idx = self._get_combination_index(key)
        item = self.table[idx]
        return item

    def __setitem__(self, key: Dict[AgentID, Sequence[PolicyID]], value: float):
        idx = self._get_combination_index(key)
        self.table[idx] = value

    def is_simulation_done(
        self, population_mapping: Dict[str, Sequence[PolicyID]]
    ) -> bool:
        """Check whether all simulations have been done"""

        idx = self._get_combination_index(population_mapping)
        return np.alltrue(self.simulation_flag[idx])

    def set_simulation_done(self, population_mapping: Dict[str, Sequence[PolicyID]]):
        idx = self._get_combination_index(population_mapping)
        self.simulation_flag[idx] = True

    def expand_table(self, pad_info):
        """Expand payoff table"""

        # head and tail

        if not any(self.table.shape):
            pad_info = [(0, 1)] * len(self.agents)
        self.table = np.pad(self.table, pad_info)
        self.simulation_flag = np.pad(self.simulation_flag, pad_info)

    def _get_combination_index(
        self, policy_combination: Dict[AgentID, Sequence[PolicyID]]
    ) -> Tuple:
        """Return combination index, if doesn't exist, expand it"""
        res = []
        expand_flag = False
        pad_info = []
        for agent in self.agents:
            idx = []
            policy_seq = policy_combination[agent]
            if isinstance(policy_seq, str):
                policy_seq = [policy_seq]

            new_policy_add_num = 0
            for p in policy_seq:
                if self._policy_idx[agent].get(p) is None:
                    expand_flag = True
                    self._policy_idx[agent][p] = len(self._policy_idx[agent])
                    new_policy_add_num += 1
                idx.append(self._policy_idx[agent][p])
            pad_info.append((0, new_policy_add_num))
            res.append(idx)
        if expand_flag:
            self.expand_table(pad_info)
        return np.ix_(*res)

    def get_combination_index(
        self, policy_combination: Dict[AgentID, Sequence[PolicyID]]
    ) -> Tuple:
        return self._get_combination_index(policy_combination)


__all__ = ["PayoffTable"]
