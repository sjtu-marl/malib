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

from types import LambdaType
from typing import List, Dict, Any, Tuple

from malib.utils.logging import Logger
from malib.utils.stopping_conditions import get_stopper
from malib.agent.manager import TrainingManager
from malib.rollout.manager import RolloutWorkerManager
from malib.common.payoff_manager import PayoffManager
from malib.common.strategy_spec import StrategySpec
from malib.scenarios import Scenario
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
        )
        self.meta_solver_type = meta_solver_type
        self.global_stopping_conditions = global_stopping_conditions


def simulate(
    rollout_manager: RolloutWorkerManager,
    strategy_specs_list: List[Dict[str, StrategySpec]],
) -> List[Tuple[Dict, Dict]]:
    rollout_manager.simulate(strategy_specs_list)
    ordered_results = rollout_manager.wait()
    # print("retrive simulation results: {}".format(ordered_results))

    # return results
    # TODO(ming): for debug, fake simulation results
    return list(zip(strategy_specs_list, ordered_results))


def execution_plan(experiment_tag: str, scenario: Scenario):
    training_manager = TrainingManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        algorithms=scenario.algorithms,
        env_desc=scenario.env_desc,
        agent_mapping_func=scenario.agent_mapping_func,
        training_config=scenario.training_config,
        log_dir=scenario.log_dir,
        remote_mode=True,
    )

    rollout_manager = RolloutWorkerManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        num_worker=scenario.num_worker,
        agent_mapping_func=scenario.agent_mapping_func,
        rollout_config=scenario.rollout_config,
        env_desc=scenario.env_desc,
        log_dir=scenario.log_dir,
    )

    payoff_manager = PayoffManager(
        agent_names=scenario.env_desc["possible_agents"].copy(),
        agent_mapping_func=scenario.agent_mapping_func,
        solve_method=scenario.meta_solver_type,
    )

    # stopper = get_stopper(scenario.global_stopping_conditions)

    equilibrium = {
        agent: {"policy-0": 1.0} for agent in scenario.env_desc["possible_agents"]
    }
    scenario.training_manager = training_manager
    scenario.rollout_manager = rollout_manager

    for i in range(10):
        Logger.info("")
        Logger.info(f"Start Global Iteration: {i}")
        scenario.prob_list_each = equilibrium

        # run best response training tasks
        info = marl_execution_plan(experiment_tag, scenario, recall_resource=False)

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
