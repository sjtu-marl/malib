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
from abc import ABC, abstractmethod

import logging


logger = logging.getLogger(__name__)


class StoppingCondition(ABC):
    @abstractmethod
    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        pass


class NoStoppingCondition(StoppingCondition):
    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        return False


class StopImmediately(StoppingCondition):
    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        return True


class RewardImprovementStopping(StoppingCondition):
    def __init__(self, mininum_reward_improvement: float) -> None:
        self.minium_reward_improvement = mininum_reward_improvement

    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        reward_this_iter = latest_trainer_result.get(
            "evaluation", {"episode_reward_mean": float("inf")}
        )["episode_reward_mean"]
        if reward_this_iter == float("inf"):
            return False
        should_stop = False
        return should_stop


class MaxIterationStopping(StoppingCondition):
    def __init__(
        self,
        max_iteration: int,
    ) -> None:
        self.max_iteration = max_iteration
        self.n_iteration = 0

    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        self.n_iteration += 1

        should_stop = False
        if self.n_iteration >= self.max_iteration:
            logger.info(
                f"Max iterations reached ({self.n_iteration}). stopping if allowed."
            )
            should_stop = True

        return should_stop


class MergeStopping(StoppingCondition):
    def __init__(self, stoppings: List[StoppingCondition]) -> None:
        super().__init__()
        self.stoppings = stoppings

    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        stops = [e.should_stop(latest_trainer_result) for e in self.stoppings]
        return all(stops)


def get_stopper(conditions: Dict[str, Any]):
    stoppings = []
    if "minimum_reward_improvement" in conditions:
        stoppings.append(
            RewardImprovementStopping(conditions["minimum_reward_improvement"])
        )
    if "max_iteration" in conditions:
        stoppings.append(MaxIterationStopping(conditions["max_iteration"]))

    if len(stoppings) == 0:
        raise NotImplementedError(f"unkonw stopping condition type: {conditions}")

    return MergeStopping(stoppings=stoppings)
