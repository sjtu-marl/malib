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

from typing import Dict, Any, List

import ray

from malib.utils.typing import AgentID
from malib.utils.logging import Logger

from malib.rollout.rolloutworker import RolloutWorker, parse_rollout_info
from malib.common.strategy_spec import StrategySpec


class PBRolloutWorker(RolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def step_rollout(
        self,
        eval_step: bool,
        strategy_specs: Dict[AgentID, StrategySpec],
        data_entrypoint_mapping: Dict[AgentID, str],
    ) -> List[Dict[str, Any]]:
        results = ray.get(
            self.env_runner.run.remote(
                rollout_config=self.rollout_config,
                strategy_specs=strategy_specs,
                data_entrypoint_mapping=data_entrypoint_mapping,
            )
        )
        # check evaluation info
        parsed_results = parse_rollout_info(results)
        Logger.debug(f"parsed results: {parsed_results}")

        return parsed_results
