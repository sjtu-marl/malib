import pytest

from malib.utils.typing import EvaluateResult
from malib.evaluator import get_evaluator, EvaluatorType
from malib.evaluator.base_evaluator import BaseEvaluator


@pytest.mark.parametrize(
    "name,stop_metrics",
    [
        (EvaluatorType.PSRO, {"stop_metrics": {"max_iteration": 100}}),
        (EvaluatorType.GENERIC, {"stop_metrics": {"max_iteration": 50}}),
    ],
)
def test_evaluator(name, stop_metrics):
    evaluator: BaseEvaluator = get_evaluator(name)(**stop_metrics)

    for i in range(stop_metrics["stop_metrics"]["max_iteration"]):
        res = evaluator.evaluate(content=None)
    assert res[EvaluateResult.CONVERGED], evaluator._iteration
