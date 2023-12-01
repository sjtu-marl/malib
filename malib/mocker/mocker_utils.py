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

from typing import Sequence, Dict, Any, Callable, List, Tuple, Union

import time


from malib.rollout.config import RolloutConfig

from malib.utils.typing import AgentID
from malib.common.strategy_spec import StrategySpec


from malib.utils.typing import PolicyID
from malib.common.payoff_manager import PayoffManager
from malib.learner.learner import Learner
from malib.backend.dataset_server.feature import BaseFeature


class FakeLearner(Learner):
    def multiagent_post_process(
        self,
        batch_info,
    ) -> Dict[str, Any]:
        pass


class FakeFeatureHandler(BaseFeature):
    pass


class FakePayoffManager(PayoffManager):
    def __init__(
        self,
        agent_names: Sequence[str],
        agent_mapping_func: Callable[[AgentID], str],
        solve_method="fictitious_play",
    ):
        pass

    def expand(self, strategy_specs: Dict[str, StrategySpec]):
        pass

    def get_matchups_eval_needed(
        self, specs_template: Dict[str, StrategySpec]
    ) -> List[Dict[str, StrategySpec]]:
        return [{}]

    def compute_equilibrium(
        self, strategy_specs: Dict[str, StrategySpec]
    ) -> Dict[str, Dict[PolicyID, float]]:
        probs = {}
        for agent, spec in strategy_specs.items():
            probs[agent] = dict(
                zip(spec.policy_ids, [1 / spec.num_policy] * spec.num_policy)
            )
        return probs

    def update_payoff(
        self, eval_data_tups: List[Tuple[Dict[str, StrategySpec], Dict[str, Any]]]
    ):
        pass


from malib.rollout.manager import RolloutWorkerManager


class FakeRolloutManager(RolloutWorkerManager):
    def __init__(
        self,
        stopping_conditions: Dict[str, Any],
        num_worker: int,
        group_info: Dict[str, Any],
        rollout_config: Union[RolloutConfig, Dict[str, Any]],
        env_desc: Dict[str, Any],
        log_dir: str,
        resource_config: Dict[str, Any] = None,
        ray_actor_namespace: str = "rollout_worker",
        verbose: bool = True,
    ):
        super().__init__(
            stopping_conditions,
            num_worker,
            group_info,
            rollout_config,
            env_desc,
            log_dir,
            resource_config,
            ray_actor_namespace,
            verbose,
        )

    def rollout(self, task_list: List[Dict[str, Any]]) -> None:
        pass

    def wait(self) -> List[Any]:
        time.sleep(0.1)

    def terminate(self):
        pass
