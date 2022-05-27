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
from typing import Dict, Any

from concurrent.futures import ThreadPoolExecutor
from malib.scenarios import Scenario

from malib.agent.manager import TrainingManager
from malib.rollout.manager import RolloutWorkerManager


class MARLScenario(Scenario):
    def __init__(
        self,
        name: str,
        log_dir: str,
        algorithms: Dict[str, Any],
        env_description: Dict[str, Any],
        rollout_config: Dict[str, Any],
        training_config: Dict[str, Any],
        agent_mapping_func: LambdaType = lambda agent: agent,
        num_worker: int = 1,
    ):
        super().__init__(name)
        self.algorithms = algorithms
        self.env_desc = env_description
        self.agent_mapping_func = agent_mapping_func
        self.training_config = training_config
        self.log_dir = log_dir
        self.num_worker = num_worker
        self.rollout_config = rollout_config


def execution_plan(scenario: Scenario):
    if hasattr(scenario, "training_manager"):
        training_manager = scenario.training_manager
    else:
        training_manager = TrainingManager(
            algorithms=scenario.algorithms,
            env_desc=scenario.env_desc,
            agent_mapping_func=scenario.agent_mapping_func,
            training_config=scenario.training_config,
            log_dir=scenario.log_dir,
            remote_mode=True,
        )

    if hasattr(scenario, "rollout_manager"):
        rollout_manager = scenario.rollout_manager
    else:
        rollout_manager = RolloutWorkerManager(
            num_worker=scenario.num_worker,
            agent_mapping_func=scenario.agent_mapping_func,
            rollout_config=scenario.rollout_config,
            env_desc=scenario.env_desc,
            log_dir=scenario.log_dir,
        )

    training_manager.add_policies(n=scenario.num_policy_each_interface)
    training_manager.run()
    rollout_manager.rollout(task_list=None)

    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(training_manager.wait)
    executor.submit(rollout_manager.wait)
    executor.shutdown(wait=True)
