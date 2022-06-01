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

from typing import Dict, Any, List, Callable

import ray

from malib.rollout.base_worker import BaseRolloutWorker, _parse_rollout_info
from malib.common.strategy_spec import StrategySpec
from malib.utils.logging import Logger


class RolloutWorker(BaseRolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def __init__(
        self,
        experiment_tag: Any,
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable,
        runtime_config: Dict[str, Any],
        log_dir: str,
        reverb_table_kwargs: Dict[str, Any],
        rollout_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        simulate_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        outer_inference_client=None,
        outer_inference_server=None,
    ):
        super().__init__(
            experiment_tag,
            env_desc,
            agent_mapping_func,
            runtime_config,
            log_dir,
            reverb_table_kwargs,
            rollout_callback,
            simulate_callback,
            outer_inference_client=outer_inference_client,
            outer_inference_server=outer_inference_server,
        )

    def step_rollout(
        self,
        eval_step: bool,
        dataserver_entrypoint: str,
        runtime_config_template: Dict[str, Any],
    ):
        tasks = [
            runtime_config_template
            for _ in range(self.worker_runtime_config["num_threads"])
        ]

        # add tasks for evaluation
        if eval_step:
            eval_runtime_config = runtime_config_template.copy()
            eval_runtime_config["flag"] = "evaluation"
            tasks.extend(
                [
                    eval_runtime_config
                    for _ in range(self.worker_runtime_config["num_eval_threads"])
                ]
            )

        Logger.debug(
            f"rollout worker step rollout with dataset: {dataserver_entrypoint}"
        )

        rets = [
            x
            for x in self.actor_pool.map(
                lambda a, task: a.run.remote(
                    agent_interfaces=self.agent_interfaces,
                    desc=task,
                    dataserver_entrypoint=dataserver_entrypoint,
                ),
                tasks,
            )
        ]

        # check evaluation info
        parsed_results = _parse_rollout_info(rets)
        Logger.debug(f"parsed results: {parsed_results}")
        return parsed_results

    def step_simulation(
        self,
        runtime_strategy_specs_list: List[Dict[str, StrategySpec]],
        runtime_config_template: Dict[str, Any],
    ):
        tasks = []
        for strategy_specs in runtime_strategy_specs_list:
            task = runtime_config_template.copy()
            task["strategy_specs"] = strategy_specs
            tasks.append(task)

        rets = [
            x
            for x in self.actor_pool.map(
                lambda a, task: a.run.remote(
                    agent_interfaces=self.agent_interfaces,
                    desc=task,
                ),
                tasks,
            )
        ]

        results = _parse_rollout_info(rets)
        return results
