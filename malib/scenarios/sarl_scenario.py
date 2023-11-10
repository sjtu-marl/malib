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
from malib.common.task import TaskType, OptimizationTask, RolloutTask

from malib.scenarios import Scenario
from malib.utils.stopping_conditions import StoppingCondition, get_stopper
from malib.utils.logging import Logger
from malib.backend.league import League
from malib.learner.manager import LearnerManager
from malib.rollout.manager import RolloutWorkerManager
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
        )
        self.num_policy_each_interface = 1
        self.resource_config = resource_config or {"training": None, "rollout": None}

    def create_global_stopper(self) -> StoppingCondition:
        return get_stopper(self.stopping_conditions)


def execution_plan(experiment_tag: str, scenario: SARLScenario, verbose: bool = True):
    # TODO(ming): simplify the initialization of training and rollout manager with a scenario instance as input
    learner_manager = LearnerManager(
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

    inference_manager = InferenceManager(
        group_info=scenario.group_info,
        ray_actor_namespace="inference_{}".format(experiment_tag),
        model_entry_point=learner_manager.learner_entrypoints,
        scenario=scenario,
    )

    rollout_manager = RolloutWorkerManager(
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

    league = League(
        learner_manager, rollout_manager, inference_manager, namespace=experiment_tag
    )

    optimization_task = OptimizationTask(
        active_agents=scenario.env_desc["possible_agents"],
        stop_conditions=scenario.stopping_conditions["training"],
    )

    strategy_specs = learner_manager.get_strategy_specs()

    rollout_task = RolloutTask(
        task_type=TaskType.ROLLOUT,
        strategy_specs=strategy_specs,
        stopping_conditions=scenario.stopping_conditions["rollout"],
        data_entrypoint_mapping=learner_manager.data_entrypoints,
    )

    evaluation_task = RolloutTask(
        task_type=TaskType.EVALUATION,
        strategy_specs=strategy_specs,
    )

    stopper = scenario.create_global_stopper()
    epoch_cnt = 0

    while True:
        rollout_results = league.submit(rollout_task, wait=True)
        training_results = league.submit(optimization_task, wait=True)
        evaluation_results = league.submit(evaluation_task, wait=True)
        epoch_cnt += 1
        if stopper.should_stop(
            evaluation_results, training_results, rollout_results, epoch_cnt
        ):
            break
        if epoch_cnt % scenario.save_interval == 0:
            league.save_checkpoint(global_step=epoch_cnt)

    results = league.get_results()
    league.terminate()

    return results
