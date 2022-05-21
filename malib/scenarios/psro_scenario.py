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

from typing import List, Dict

from malib.utils.typing import AgentID
from malib.utils.stopping_conditions import get_stopper
from malib.agent.manager import TrainingManager
from malib.rollout.manager import RolloutWorkerManager
from malib.common.payoff_manager import PayoffManager
from malib.common.strategy_spec import StrategySpec
from malib.scenarios import Scenario
from malib.scenarios.marl_scenario import execution_plan as marl_execution_plan


def simulate(
    rollout_manager: RolloutWorkerManager,
    strategy_specs_list: List[Dict[str, StrategySpec]],
):
    rollout_manager.simulate(strategy_specs_list)
    results = rollout_manager.wait()
    return results


def execution_plan(scenario: Scenario):
    training_manager = TrainingManager(
        algorithms=scenario.algorithms,
        env_desc=scenario.env_desc,
        interface_config=scenario.interface_config,
        agent_mapping_func=scenario.agent_mapping_func,
        training_config=scenario.training_config,
        log_dir=scenario.log_dir,
        remote_mode=True,
    )

    rollout_manager = RolloutWorkerManager(
        num_worker=scenario.num_worker,
        agent_mapping_func=scenario.agent_mapping_func,
        rollout_configs=scenario.rollout_configs,
        env_desc=scenario.env_desc,
        log_dir=scenario.log_dir,
    )

    payoff_manager = PayoffManager(
        agent_names=scenario.env_desc["possible_agents"],
        solve_method=scenario.meta_solver_type,
    )

    stopper = get_stopper(scenario.global_stopping_conditions)

    equilibrium = {
        agent: {"policy-0": 1.0} for agent in scenario.env_desc["possible_agents"]
    }
    scenario.training_manager = training_manager
    scenario.rollout_manager = rollout_manager

    # TODO(ming): eval based on the exploitability
    while stopper.should_stop():
        scenario.prob_list_each = equilibrium
        info = marl_execution_plan(scenario)
        # extend payoff tables with new brs
        strategy_specs: Dict[AgentID, StrategySpec] = info["strategy_specs"]
        payoff_manager.expand(brs=strategy_specs)

        # retrieve specs list, a dict as a joint strategy spec
        strategy_specs_list = payoff_manager.get_pending_matchups(strategy_specs)
        results = simulate(rollout_manager, strategy_specs_list=strategy_specs_list)
        payoff_manager.update_payoff(results)

        # update probs
        equilibrium = payoff_manager.compute_equilibrium(strategy_specs)
