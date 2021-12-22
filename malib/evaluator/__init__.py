from typing import Callable

from .base_evaluator import BaseEvaluator as Evaluator
from .psro import PSROEvaluator
from .generic import GenericEvaluator


class EvaluatorType:
    PSRO = "psro"
    GENERIC = "generic"
    SIMPLE = "simple"

    def __repr__(self):
        return "<EvaluatorType: [psro, generic, simple]>"


def get_evaluator(name: str) -> Callable:
    if name == EvaluatorType.PSRO:
        return PSROEvaluator
    elif name == EvaluatorType.GENERIC:
        return GenericEvaluator
    else:
        raise ValueError(
            f"Expected evaluator type: {EvaluatorType}, while the input is: {name}"
        )
