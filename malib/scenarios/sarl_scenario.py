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

from typing import Dict, Any
from malib.common.task import OptimizationTask, RolloutTask

from malib.scenarios import Scenario

from malib.utils.logging import Logger
from malib.backend.league import League
from malib.learner.manager import TrainingManager
from malib.rollout.manager import RolloutWorkerManager, TaskType
from malib.rollout.inference.manager import InferenceManager


class SARLScenario(Scenario):
    def __init__(
        self,
        name: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Any],
        training_config: Dict[str, Any],
        rollout_config: Dict[str, Any],
        stopping_conditions: Dict[str, Any],
        dataset_config: Dict[str, Any],
        parameter_server_config: Dict[str, Any],
        resource_config: Dict[str, Any] = None,
    ):
        super().__init__(
            name,
            log_dir,
            env_desc,
            algorithms,
            lambda agent: "default",
            training_config,
            rollout_config,
            stopping_conditions,
            dataset_config,
            parameter_server_config,
        )
        self.num_policy_each_interface = 1
        self.resource_config = resource_config or {"training": None, "rollout": None}


def execution_plan(experiment_tag: str, scenario: SARLScenario, verbose: bool = True):
    # TODO(ming): simplify the initialization of training and rollout manager with a scenario instance as input
    training_manager = TrainingManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        algorithms=scenario.algorithms,
        env_desc=scenario.env_desc,
        agent_mapping_func=scenario.agent_mapping_func,
        group_info=scenario.group_info,
        training_config=scenario.training_config,
        log_dir=scenario.log_dir,
        remote_mode=True,
        resource_config=scenario.resource_config["training"],
        ray_actor_namespace="learner_{}".format(experiment_tag),
        verbose=verbose,
    )

    rollout_manager = RolloutWorkerManager(
        experiment_tag=experiment_tag,
        stopping_conditions=scenario.stopping_conditions,
        num_worker=scenario.num_worker,
        group_info=scenario.group_info,
        rollout_config=scenario.rollout_config,
        env_desc=scenario.env_desc,
        log_dir=scenario.log_dir,
        resource_config=scenario.resource_config["rollout"],
        ray_actor_namespace="rollout_{}".format(experiment_tag),
        verbose=verbose,
    )

    inference_manager = InferenceManager(
        group_info=scenario.group_info,
        ray_actor_namespace="inference_{}".format(experiment_tag),
        entrypoints=training_manager.get_data_entrypoints(),
        scenario=scenario,
    )

    league = League(rollout_manager, training_manager, inference_manager)

    # NOTE(ming): if all agents are active, the strategy specs should not contain any pids
    strategy_specs = training_manager.add_policies(n=1)
    Logger.info(
        f"Training manager was inistialized with a strategy spec:\n{strategy_specs}"
    )

    optimization_task = OptimizationTask(
        active_agents=scenario.env_desc["possible_agents"],
        stop_conditions=scenario.stopping_conditions["training"],
    )
    training_manager.submit(optimization_task)

    rollout_task = RolloutTask(
        task_type=TaskType.ROLLOUT,
        strategy_specs=strategy_specs,
        stopping_conditions=scenario.stopping_conditions["rollout"],
        data_entrypoint_mapping=training_manager.get_data_entrypoint_mapping(),
    )

    evaluation_task = RolloutTask(
        task_type=TaskType.EVALUATION,
        strategy_specs=strategy_specs,
    )

    rollout_manager.submit(rollout_task)
    rollout_manager.submit(evaluation_task)

    results = league.get_results()
    league.terminate()

    return results
