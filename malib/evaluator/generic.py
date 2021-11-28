from typing import Dict, Any

from malib.utils.typing import EvaluateResult
from malib.evaluator.base_evaluator import BaseEvaluator


class GenericEvaluator(BaseEvaluator):
    class StopMetrics:
        MAX_ITERATION = "max_iteration"

    def __init__(self, **config):
        super(GenericEvaluator, self).__init__(
            config.get("stop_metrics", {}), name="generic"
        )
        self._iteration = 0

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        res = EvaluateResult.default_result()
        self._iteration += 1
        res[EvaluateResult.CONVERGED] = (
            self._metrics.get(GenericEvaluator.StopMetrics.MAX_ITERATION, 1)
            == self._iteration
        )
        return res
