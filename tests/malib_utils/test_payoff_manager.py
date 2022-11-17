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

import pytest
import numpy as np

from malib.common.strategy_spec import StrategySpec
from malib.common.payoff_manager import (
    PayoffManager,
    DefaultSolver,
    PayoffTable,
    SimulationFlag,
)


@pytest.mark.parametrize("solve_method", ["fictitious_play", "alpharank"])
def test_default_solver(solve_method: str):
    solver = DefaultSolver(solve_method=solve_method)
    a = np.random.random((4, 4))
    b = -a
    solver.solve(payoffs_seq=[a, b])


@pytest.mark.parametrize("n_player", [2, 4])
def test_payoff_table(n_player: int):

    agents = [f"player_{i}" for i in range(n_player)]
    # start from one policy each player
    shape = [0] * n_player
    simulation_flag = SimulationFlag(np.zeros(shape).astype(bool))

    table = PayoffTable(
        identify=agents[0], agents=agents, shared_simulation_flag=simulation_flag
    )

    pad_info = [(0, 1)] * n_player
    table.expand_table(pad_info)
    # policy_combination =

    # table.get_combination_index(policy_combination=policy_combination)


@pytest.mark.parametrize("n_players", [2, 4])
@pytest.mark.parametrize("solve_method", ["fictitious_play", "alpharank"])
class TestPayoffManager:
    @pytest.fixture(autouse=True)
    def init(self, n_players: int, solve_method: str):
        agents = [f"player_{i}" for i in range(n_players)]

        manager = PayoffManager(
            agent_names=agents,
            agent_mapping_func=lambda agent: agent,
            solve_method=solve_method,
        )

        self.manager = manager
        self.agents = agents

    def test_table_expand(self):
        strategy_specs = {
            agent: StrategySpec(
                identifier=agent,
                policy_ids=["policy_0"],
                meta_data={"policy_cls": None, "kwargs": None, "experiment_tag": None},
            )
            for agent in self.agents
        }
        self.manager.expand(strategy_specs)
        return strategy_specs

    def test_matchup_retrive(self):
        strategy_specs = {
            agent: StrategySpec(
                identifier=agent,
                policy_ids=["policy_0"],
                meta_data={"policy_cls": None, "kwargs": None, "experiment_tag": None},
            )
            for agent in self.agents
        }
        matchups = self.manager.get_matchups_eval_needed(strategy_specs)

    def test_check_done(self):
        # check done
        strategy_specs = self.test_table_expand()
        population_mapping = {agent: ["policy_0"] for agent in self.agents}
        all_done = self.manager.check_done(population_mapping)
        assert not all_done

    def test_payoff_updates(self):
        strategy_specs = self.test_table_expand()
        eval_results = {
            "evaluation": {f"agent_reward/{agent}_mean": 1.0 for agent in self.agents}
        }
        eval_data_tups = [(strategy_specs, eval_results)]
        self.manager.update_payoff(eval_data_tups)

        # retrieve payoff tables to check the consistency
        payoffs = list(self.manager.payoffs.values())
        matrices = [e.table for e in payoffs]

        ref = matrices[0]

        for data in matrices[1:]:
            assert np.all(data == ref), (data, ref)

        return strategy_specs

    def test_aggregation(self):
        # aggregate with fake equilibrium
        self.test_payoff_updates()
        equilibrium = {agent: {"policy_0": 1.0} for agent in self.agents}
        brs = None
        res = self.manager.aggregate(equilibrium, brs)
        for v in res.values():
            assert v == 1.0, v

        # expand with new policy
        strategy_specs = {
            agent: StrategySpec(
                identifier=agent,
                policy_ids=["policy_1"],
                meta_data={"policy_cls": None, "kwargs": None, "experiment_tag": None},
            )
            for agent in self.agents
        }
        self.manager.expand(strategy_specs)
        brs = {agent: "policy_1" for agent in self.agents}
        res = self.manager.aggregate(equilibrium, brs)
        for v in res.values():
            assert v == 0.0, v

    def test_compute_equilibrium(self):
        strategy_specs = self.test_payoff_updates()
        eq = self.manager.compute_equilibrium(strategy_specs)
        for v in eq.values():
            assert len(v) == 1
            assert v["policy_0"] == 1.0, v

        pmapping = {agent: list(v.keys()) for agent, v in eq.items()}
        self.manager.update_equilibrium(pmapping, eq)

        eq_check = self.manager.get_equilibrium(pmapping)

        for agent, pdist in eq.items():
            for k, v in pdist.items():
                assert eq_check[agent][k] == v, eq_check[agent]
