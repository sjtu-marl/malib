"""
Training evaluator controls the training stopping
"""
from typing import Dict, Any

from malib.evaluator.base_evaluator import BaseEvaluator


class Simple(BaseEvaluator):
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        pass
