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

from collections import defaultdict
from types import LambdaType
from typing import List, Dict, Any, Tuple
from pprint import pformat

import ray

from malib.utils.logging import Logger
from malib.utils.stopping_conditions import get_stopper
from malib.utils.exploitability import measure_exploitability
from malib.agent.manager import TrainingManager
from malib.rollout.manager import RolloutWorkerManager
from malib.common.payoff_manager import PayoffManager
from malib.common.strategy_spec import StrategySpec
from malib.scenarios.marl_scenario import (
    execution_plan as marl_execution_plan,
    MARLScenario,
)


class PSROScenario(MARLScenario):
    def __init__(
        self,
        name: str,
        log_dir: str,
        algorithms: Dict[str, Any],
        env_description: Dict[str, Any],
        rollout_config: Dict[str, Any],
        training_config: Dict[str, Any],
        global_stopping_conditions: Dict[str, Any],
        meta_solver_type: str = "fictitious_play",
        agent_mapping_func: LambdaType = ...,
        num_worker: int = 1,
        stopping_conditions: Dict[str, Any] = None,
        dataset_config: Dict[str, Any] = None,
        parameter_server_config: Dict[str, Any] = None,
        resource_config: Dict[str, Any] = None,
    ):
        """Construct a learning scenario for Policy Space Response Oracle methods.

        Args:
            name (str): Scenario name, for experiment tag creating and identification.
            log_dir (str): Log directory.
            algorithms (Dict[str, Any]): A dict that provides a series of algorithms, must indludes algorithm named with `default`.
            env_description (Dict[str, Any]): Environment description.
            rollout_config (Dict[str, Any]): Rollout configuration.
            training_config (Dict[str, Any]): Training configuration, for the construction of `AgentInterface`.
            global_stopping_conditions (Dict[str, Any]): Global stopping conditions to control the outer loop.
            meta_solver_type (str): Meta solver type, `fictitious_play` or `alpharank`.
            agent_mapping_func (LambdaType, optional): Agent mapping function, maps from environment agents to runtime ids, all workers share the same mapping func. Defaults to lambdaagent:agent.
            num_worker (int, optional): Indicates how many `RolloutWorker` will be initialized. Defaults to 1.
            stopping_conditions (Dict[str, Any], optional): Stopping conditions, should contain `rollout` and `training`. Defaults to None.
            dataset_config (Dict[str, Any], optional): Dataset configuration. Defaults to None.
            parameter_server_config (Dict[str, Any], optional): Parameter server configuration. Defaults to None.
        """

        super().__init__(
            name,
            log_dir,
            algorithms,
            env_description,
            rollout_config,
            training_config,
            agent_mapping_func,
            num_worker,
            stopping_conditions,
            dataset_config,
            parameter_server_config,
            resource_config,
        )
        self.meta_solver_type = meta_solver_type
        self.global_stopping_conditions = global_stopping_conditions


def simulate(
    rollout_manager: RolloutWorkerManager,
    strategy_specs_list: List[Dict[str, StrategySpec]],
) -> List[Tuple[Dict, Dict]]:
    """Run simulations for a list of strategy spec dict. One for each policy combination.

    Args:
        rollout_manager (RolloutWorkerManager): Rollout manager instance.
        strategy_specs_list (List[Dict[str, StrategySpec]]): A list of strategy spec dicts.

    Returns:
        List[Tuple[Dict, Dict]]: A list of tuple that composes of strategy spec dict and the corresponding evaluation results.
    """

    rollout_manager.simulate(strategy_specs_list)
    ordered_results = rollout_manager.wait()
    return list(zip(strategy_specs_list, ordered_results))


def execution_plan(experiment_tag: str, scenario: PSROScenario):
    """Execution plan for running PSRO scenario.

    Args:
        experiment_tag (str): Experiment identifier.
        scenario (PSROScenario): PSRO Scenario instance.
    """

    training_manager = TrainingManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        algorithms=scenario.algorithms,
        env_desc=scenario.env_desc,
        agent_mapping_func=scenario.agent_mapping_func,
        training_config=scenario.training_config,
        log_dir=scenario.log_dir,
        remote_mode=True,
        resource_config=scenario.resource_config["training"],
        verbose=False,
    )

    rollout_manager = RolloutWorkerManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        num_worker=scenario.num_worker,
        agent_mapping_func=scenario.agent_mapping_func,
        rollout_config=scenario.rollout_config,
        env_desc=scenario.env_desc,
        log_dir=scenario.log_dir,
        resource_config=scenario.resource_config["rollout"],
        verbose=False,
    )

    payoff_manager = PayoffManager(
        agent_names=scenario.env_desc["possible_agents"].copy(),
        agent_mapping_func=scenario.agent_mapping_func,
        solve_method=scenario.meta_solver_type,
    )

    stopper = get_stopper(scenario.global_stopping_conditions)

    equilibrium = {
        agent: {"policy-0": 1.0} for agent in scenario.env_desc["possible_agents"]
    }
    scenario.training_manager = training_manager
    scenario.rollout_manager = rollout_manager

    i = 0
    while True:
        Logger.info("")
        Logger.info(f"Global Iteration: {i}")
        scenario.prob_list_each = equilibrium

        # run best response training tasks
        info = marl_execution_plan(
            experiment_tag, scenario, recall_resource=False, verbose=False
        )

        # extend payoff tables with brs
        strategy_specs: Dict[str, StrategySpec] = info["strategy_specs"]
        payoff_manager.expand(strategy_specs=strategy_specs)

        # retrieve specs list, a dict as a joint strategy spec
        eval_matchups = payoff_manager.get_matchups_eval_needed(
            specs_template=strategy_specs
        )
        # retrive strategy spec dict for which cell not be evaluated yet.
        eval_data_tups = simulate(rollout_manager, strategy_specs_list=eval_matchups)
        payoff_manager.update_payoff(eval_data_tups)

        # update probs
        equilibrium = payoff_manager.compute_equilibrium(strategy_specs)
        Logger.info("\tequilibrium: {}".format(pformat(equilibrium)))

        # run evaluation
        populations = defaultdict(dict)
        for agent, strategy_spec in strategy_specs.items():
            for spec_policy_id in strategy_spec.policy_ids:
                policy = strategy_spec.gen_policy()
                info = ray.get(
                    scenario.parameter_server.get_weights.remote(
                        spec_id=strategy_spec.id,
                        spec_policy_id=spec_policy_id,
                    )
                )
                policy.load_state_dict(info["weights"])
                populations[agent][spec_policy_id] = policy

        populations = dict(populations)
        nash_conv = measure_exploitability(
            scenario.env_desc["config"]["env_id"], populations, equilibrium
        )
        Logger.info(f"\tnash_conv: {nash_conv.nash_conv}")
        i += 1

        if stopper.should_stop(None):
            break
