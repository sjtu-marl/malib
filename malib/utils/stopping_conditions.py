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


class RewardAndIterationStopping(StoppingCondition):
    def __init__(
        self,
        max_iteration: int,
        minimum_reward_improvement: float,
    ) -> None:
        self.max_iteration = max_iteration
        self.minimum_reward_imp = minimum_reward_improvement
        self.n_iteration = 0

    def should_stop(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        self.n_iteration += 1
        br_reward_this_iter = latest_trainer_result.get(
            "evaluation", {"episode_reward_mean": float("inf")}
        )["episode_reward_mean"]
        if br_reward_this_iter == float("inf"):
            return False

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
    if "reward_and_iteration":
        stoppings.append(
            RewardAndIterationStopping(**conditions["reward_and_iteration"])
        )

    if len(stoppings) == 0:
        raise NotImplementedError(f"unkonw stopping condition type: {conditions}")

    return MergeStopping(stoppings=stoppings)
