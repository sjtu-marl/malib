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

from malib.rollout.rolloutworker import RolloutWorker, parse_rollout_info
from malib.utils.logging import Logger


class PBRolloutWorker(RolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def step_rollout(
        self,
        eval_step: bool,
        rollout_config: Dict[str, Any],
        data_entrypoint_mapping: Dict[str, Any],
    ):
        tasks = [rollout_config for _ in range(self.rollout_config["num_threads"])]

        # add tasks for evaluation
        if eval_step:
            eval_runtime_config = rollout_config.copy()
            eval_runtime_config["flag"] = "evaluation"
            tasks.extend(
                [
                    eval_runtime_config
                    for _ in range(self.rollout_config["num_eval_threads"])
                ]
            )

        rets = [
            x
            for x in self.env_runner_pool.map(
                lambda a, task: a.run.remote(
                    inference_clients=self.inference_clients,
                    rollout_config=task,
                    data_entrypoint_mapping=data_entrypoint_mapping,
                ),
                tasks,
            )
        ]

        # check evaluation info
        parsed_results = parse_rollout_info(rets)
        Logger.debug(f"parsed results: {parsed_results}")
        return parsed_results
