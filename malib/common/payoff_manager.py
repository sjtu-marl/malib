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

from typing import List, Union, Sequence, Dict, Tuple, Any, Callable
import copy
import itertools
import logging

import nashpy as nash
import numpy as np

from dataclasses import dataclass

from malib.utils.typing import AgentID, PolicyID
from malib.utils.logging import Logger
from malib.common.strategy_spec import StrategySpec

try:
    from open_spiel.python.egt import alpharank, utils as alpharank_utils
except ImportError as e:
    logging.warning(
        "Cannot import alpharank utils, if you wanna run meta game experiments, please install open_spiel before that."
    )


class DefaultSolver:
    """A Solver to find certain solution concept, e.g. nash equilibrium."""

    def __init__(self, solve_method="fictitious_play"):
        """Initialze the solver

        :param solve_method: a string to tell which solve method should be used, "fictious_play" or "alpharank",default="fictitous_play".
        """
        self._solve_method = solve_method

    def fictitious_play(self, payoffs_seq):
        """solve the game with fictitious play, only suppoort 2-player games

        :param payoffs_seq: a sequence of the game's payoff matrix, which can be of length one or two, when of length one, just as take [M, -M] as input
        :return: the nash equilirium computed by fictious play, which order is corresponding to *payoff_seq*
        """
        game = nash.Game(*payoffs_seq)

        *_, eqs = iter(game.fictitious_play(iterations=10000))
        eqs = [tuple(map(lambda x: x / np.sum(x), eqs))]
        return eqs[0]

    def alpharank(self, payoffs_seq):
        """Use alpharank to solve the game, for more details, you can check https://github.com/deepmind/open_spiel/blob/master/docs/alpha_rank.md

        :param payoffs_seq: a sequence of empirical payoffs
        :return: the solution computed by alpharank, which is a sequnce of np.ndarray of probability in each population
        """

        def remove_epsilon_negative_probs(probs, epsilon=1e-9):
            """Removes negative probabilities that occur due to precision errors."""
            if len(probs[probs < 0]) > 0:  # pylint: disable=g-explicit-length-test
                # Ensures these negative probabilities aren't large in magnitude, as that is
                # unexpected and likely not due to numerical precision issues
                assert np.alltrue(
                    np.min(probs[probs < 0]) > -1.0 * epsilon
                ), "Negative Probabilities received were: {}".format(probs[probs < 0])

                probs[probs < 0] = 0
                probs = probs / np.sum(probs)
            return probs

        def get_alpharank_marginals(payoff_tables, pi):
            """Returns marginal strategy rankings for each player given joint rankings pi.

            Args:
              payoff_tables: List of meta-game payoff tables for a K-player game, where
                each table has dim [n_strategies_player_1 x ... x n_strategies_player_K].
                These payoff tables may be asymmetric.
              pi: The vector of joint rankings as computed by alpharank. Each element i
                corresponds to a unique integer ID representing a given strategy profile,
                with profile_to_id mappings provided by
                alpharank_utils.get_id_from_strat_profile().

            Returns:
              pi_marginals: List of np.arrays of player-wise marginal strategy masses,
                where the k-th player's np.array has shape [n_strategies_player_k].
            """
            num_populations = len(payoff_tables)

            if num_populations == 1:
                return pi
            else:
                num_strats_per_population = (
                    alpharank_utils.get_num_strats_per_population(
                        payoff_tables, payoffs_are_hpt_format=False
                    )
                )
                num_profiles = alpharank_utils.get_num_profiles(
                    num_strats_per_population
                )
                pi_marginals = [np.zeros(n) for n in num_strats_per_population]
                for i_strat in range(num_profiles):
                    strat_profile = alpharank_utils.get_strat_profile_from_id(
                        num_strats_per_population, i_strat
                    )
                    for i_player in range(num_populations):
                        pi_marginals[i_player][strat_profile[i_player]] += pi[i_strat]
                return pi_marginals

        joint_distr = alpharank.sweep_pi_vs_epsilon(payoffs_seq)
        joint_distr = remove_epsilon_negative_probs(joint_distr)
        marginals = get_alpharank_marginals(payoffs_seq, joint_distr)

        return marginals

    def solve(self, payoffs_seq):
        if self._solve_method == "fictitious_play" and len(payoffs_seq) == 2:
            return self.fictitious_play(payoffs_seq)
        elif self._solve_method == "alpharank":  # when number of players > 2
            return self.alpharank(payoffs_seq)


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


class PayoffManager:
    def __init__(
        self,
        agent_names: Sequence[str],
        agent_mapping_func: Callable,
        solve_method="fictitious_play",
    ):
        """Construct a payoff manager.

        Args:
            agent_names (Sequence[str]): a sequence of names which indicate players in the game
            solve_method (str, optional): The method used to solve the game, "fictitious_play" or "alpharank". Defaults to "fictitious_play".
        """

        self.agents = agent_names
        self.agent_mapping_func = agent_mapping_func
        self.num_player = len(agent_names)
        self.solver = DefaultSolver(solve_method)

        # a map for each player in which is a list
        self._policy = {an: [] for an in agent_names}
        self._policy_idx = {an: {} for an in agent_names}
        self._policy_config = {an: [] for an in agent_names}

        # table for each player
        self._payoff_tables = {
            agent: PayoffTable(agent, self.agents) for agent in self.agents
        }

        # a list store equilibria, in which is a dict of the
        #  population distribution of each player
        self._equilibrium = {}

    @property
    def payoffs(self) -> Dict[AgentID, PayoffTable]:
        """Return a copy of payoff tables.

        Returns:
            Dict[AgentID, PayoffTable]: A dict of payoff tables.
        """

        return self._payoff_tables.copy()

    @property
    def equilibrium(self):
        return self._equilibrium

    def expand(self, strategy_specs: Dict[str, StrategySpec]):
        agent_pids = {}
        for agent in self.agents:
            rid = self.agent_mapping_func(agent)
            # TODO(ming): consider only the latest active policy now
            agent_pids[agent] = strategy_specs[rid].policy_ids[-1]

        # forced expand with value
        for agent in self.agents:
            self._payoff_tables[agent][agent_pids] = 0.0

    def check_done(self, population_mapping: Dict):
        """Check whether all payoff values have been updated, a population_mapping
        will be hashed as a key to retrieve the simulation status table shared by
        related agents.

        :param Dict population_mapping: a dict of (agent_name, policy
        """

        # XXX(ming): another more efficient method is to check simulation done with
        #  sub-matrix extraction
        #  >>> policy_comb_idx = self._get_combination_index(policy_mapping)
        #  >>> all_done = np.alltrue(simulation[policy_comb_idx])
        all_done = True
        for agent in population_mapping.keys():
            all_done = self._payoff_tables[agent].is_simulation_done(population_mapping)
            if not all_done:
                break

        return all_done

    def aggregate(
        self,
        equilibrium: Dict[AgentID, Dict[PolicyID, float]],
        brs: Dict[AgentID, PolicyID] = None,
    ) -> Dict[AgentID, float]:
        """Return weighted or nash payoff value"""

        res = {agent_id: 0.0 for agent_id in equilibrium}
        population_combination = {
            agent: list(e.keys()) for agent, e in equilibrium.items()
        }

        # retrieve partial payoff matrix
        if brs is None:
            res = {
                agent: self._payoff_tables[agent][population_combination]
                for agent in self.agents
            }  # self.get_selected_table(population_combination)
        else:
            # m*m*...*1*...*m
            for agent in brs.keys():
                tmp_comb = copy.copy(population_combination)
                # temporary replace the population of the ego agent
                # for computing the weighted payoff value: trainable policy vs. other agents
                tmp_comb[agent] = [brs[agent]]
                res[agent] = self._payoff_tables[agent][
                    tmp_comb
                ]  # self.get_selected_table(tmp_comb)

        # then aggregate the payoff matrix along axis
        weight_vectors = [
            np.asarray([list(equilibrium[agent].values())]) for agent in self.agents
        ]

        if brs is None:
            # in case of computing nash values
            weight_mat = np.asarray([[1.0]])
            for vector in weight_vectors:
                weight_mat = np.einsum("ij,j...->i...", vector.T, weight_mat)
                weight_mat = np.expand_dims(weight_mat, axis=0)
            weight_mat = np.squeeze(weight_mat, axis=0)
            weight_mat = np.squeeze(weight_mat, axis=-1)
            weight_mat = np.transpose(weight_mat)
            for agent in self.agents:
                assert weight_mat.shape == res[agent].shape, (
                    weight_mat.shape,
                    res[agent].shape,
                    equilibrium[agent],
                )
                res[agent] = (res[agent] * weight_mat).sum()
        else:
            # in case of computing
            # weight_mat = np.asarray([[1.0]])
            for agent in brs.keys():
                # ignore this one
                tmp = np.asarray([[1.0]])
                agent_axis = self.agents.index(agent)
                for i, vector in enumerate(weight_vectors):
                    if i == agent_axis:
                        continue
                    tmp = np.einsum("ij,j...->i...", vector.T, tmp)
                    tmp = np.expand_dims(tmp, axis=0)
                tmp = np.squeeze(tmp, axis=-1)
                tmp = np.squeeze(tmp, axis=0)
                tmp = np.transpose(tmp)
                tmp = np.expand_dims(tmp, axis=agent_axis)
                assert tmp.shape == res[agent].shape, (
                    tmp.shape,
                    res[agent].shape,
                    equilibrium[agent],
                    i,
                    tmp_comb,
                    agent,
                )
                res[agent] = (res[agent] * tmp).sum()
                # weight_mat = np.einsum("ij,j...->i...", weight_vectors[i].T, weight_mat)
                # weight_mat = np.expand_dims(weight_mat, axis=0)

        return res

    def update_payoff(self, eval_data: List[Tuple[Dict, Dict]]):
        """Update the payoff table, and set the corresponding simulation_flag to True"""

        raise NotImplementedError

        # population_combination = content.policy_combination
        # # for agent in self.agents:
        # for k, v in content.statistics.items():
        #     # Logger.debug("get kd: {} {}".format(k, v))
        #     ks = k.split("/")
        #     if "reward" == ks[-2] and "mean" in ks[-1]:
        #         agent = ks[-1][:-5]
        #         self._payoff_tables[agent][population_combination] = v
        #         self._payoff_tables[agent].set_simulation_done(population_combination)

    def compute_equilibrium(
        self, strategy_specs: Dict[str, StrategySpec]
    ) -> Dict[str, Dict[PolicyID, float]]:
        """Compute equilibrium of given strategy specs.

        Args:
            strategy_specs (Dict[str, StrategySpec]): A dict of strategy specs.

        Returns:
            Dict[str, Dict[PolicyID, float]]: _description_
        """

        # sub_payoff_matrix = self.get_selected_table(population_combination)
        # map strategy specs to agent specs
        population_mapping = {}
        for agent in self.agents:
            rid = self.agent_mapping_func(agent)
            population_mapping[agent] = list(strategy_specs[rid].policy_ids)

        sub_payoff_matrix = [
            self._payoff_tables[agent][population_mapping] for agent in self.agents
        ]

        if sub_payoff_matrix[0].shape[-1] == 1:
            res = {
                agent: dict(zip(p, [1 / max(1, len(p))] * len(p)))
                for agent, p in population_mapping.items()
            }
        else:
            eps = self.solver.solve(sub_payoff_matrix)
            dist = [e.tolist() for e in eps]

            res = {
                agent: dict(zip(p, dist[i]))
                for i, (agent, p) in enumerate(population_mapping.items())
            }

        # convert agent eq to runtime eq, consider only one to one.
        runtime_probs = {}
        for agent, probs in res.items():
            rid = self.agent_mapping_func(agent)
            runtime_probs[rid] = probs

        return res

    def update_equilibrium(
        self,
        population_mapping: Dict[PolicyID, Sequence[PolicyID]],
        eqbm: Dict[PolicyID, Dict[PolicyID, float]],
    ):
        """Update the equilibrium of certain population mapping in the payoff table
        :param Dict[PolicyID,Sequence[PolicyID]] population_mapping: a dict from agent_name to a sequence of policy ids
        :param Dict[PolicyID,Dict[PolicyID,float]] eqbm: the nash equilibrium which is a dict from agent_name to a dict from policy id to float
        """
        hash_key = self._hash_population_mapping(population_mapping)
        self._equilibrium[hash_key] = eqbm.copy()

    def get_equilibrium(
        self, population_mapping: Dict[AgentID, Sequence[PolicyID]]
    ) -> Dict[AgentID, Dict[PolicyID, Union[float, np.ndarray]]]:
        """Get the equilibrium stored in the payoff manager

        :param Dict[AgentID,Sequence[PolicyID]] population_mapping: a dict from agent_name to a sequence of policy ids
        :return: the nash equilibrium which is a dict from agent_name to a dict from policy id to float
        >>> eqbm = {"player_0": {"policy_0": 1.0, "policy_1": 0.0}, "player_1": {"policy_0": 0.3, "policy_1": 0.7}}
        >>> population_mapping = {"player_0": ["policy_0", "policy_1"], "player_1": ["policy_0", "policy_1"]}
        >>> self.update_equilibrium(population_mapping, eqbm)
        >>> self.get_equilibrium(population_mapping)
        ... {"player_0": {"policy_0": 1.0, "policy_1": 0.0}, "player_1": {"policy_0": 0.3, "policy_1": 0.7}}
        """

        hash_key = self._hash_population_mapping(population_mapping)
        agent = list(population_mapping.keys())[0]
        assert hash_key in self._equilibrium, (
            hash_key,
            self._equilibrium.keys(),
            self._payoff_tables[agent].table.shape,
            self._payoff_tables[agent].table,
        )
        eq = self._equilibrium[hash_key]

        return eq.copy()

    def _hash_population_mapping(
        self, population_mapping: Dict[PolicyID, Sequence[PolicyID]]
    ) -> str:
        """
        currently make it to a string
        """
        sorted_mapping = {}
        ans = ""
        for an in self.agents:
            ans += an + ":"
            sorted_mapping[an] = sorted(population_mapping[an])
            for pid in sorted_mapping[an]:
                ans += pid + ","
            ans += ";"
        return ans

    def _get_pending_matchups(
        self, agent_name: AgentID, policy_id: PolicyID, policy_config: Dict[str, Any]
    ) -> List[Dict]:
        """Generate match description with policy combinations"""

        agent_policy_list = []
        for an in self.agents:
            if an == agent_name:
                agent_policy_list.append([(policy_id, policy_config)])
            else:
                # skip empty policy
                if len(self._policy[an]) == 0:
                    continue
                # append all other agent policies
                agent_policy_list.append(
                    list(zip(self._policy[an], self._policy_config[an]))
                )

        # if other agents has no available policies, return an empty list
        if len(agent_policy_list) < len(self.agents):
            return []

        pending_comb_list = [comb for comb in itertools.product(*agent_policy_list)]
        return [
            {an: pending_comb[i] for i, an in enumerate(self.agents)}
            for pending_comb in pending_comb_list
        ]

    def get_pending_matchups(
        self, agent_name: AgentID, policy_id: PolicyID, policy_config: Dict[str, Any]
    ) -> List[Dict]:
        """Add a new policy for an agent and get the needed matches.

        :param AgentID agent_name: the agent name for which a new policy will be added
        :param PolicyID policy_id: the policy to be added
        :param Dict[str,Any] policy_config: the config of the added policy
        :return: a list of match combinations, each is a dict from agent_name to a tuple of policy_id and policy_config
        """
        if policy_id in self._policy[agent_name]:
            return []

        # May have some problems for concurrent version, but we have no demand for a concurrent payoff table ...
        self._policy_idx[agent_name][policy_id] = len(self._policy[agent_name])
        self._policy[agent_name].append(policy_id)
        self._policy_config[agent_name].append(policy_config)

        return self._get_pending_matchups(agent_name, policy_id, policy_config)
