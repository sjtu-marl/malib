import abc
import numpy as np
from malib.utils.typing import Tuple, Any, Union, Callable


class BaseAggregator(abc.ABC):
    def __init__(self, name: str):
        self._name = name

    @abc.abstractmethod
    def apply(self, x):
        raise NotImplementedError


class Mean(BaseAggregator):
    def __init__(
        self, weights: Tuple[float] = None, scale: int = None, *args, **kwargs
    ):
        super().__init__(name="mean")
        self._scale = scale
        self._weights = np.array(weights) if weights else None

    def apply(self, x, *args: Any, **kwargs: Any) -> Any:
        if self._weights:
            x = np.multiply(np.array(x), self._weights)
        return (
            np.divide(np.sum(x), self._scale, *args, **kwargs)
            if self._scale
            else np.mean(x, *args, **kwargs)
        )


class Max(BaseAggregator):
    def __init__(self, weights: Tuple[float] = None, *args, **kwargs):
        super().__init__(name="max")
        self._weights = np.array(weights) if weights else None

    def apply(self, x, *args: Any, **kwargs: Any) -> Any:
        if self._weights:
            x = np.multiply(np.array(x), self._weights)
        return np.max(x, *args, **kwargs)


class Min(BaseAggregator):
    def __init__(self, weights: Tuple[float] = None, *args, **kwargs):
        super().__init__(name="min")
        self._weights = np.array(weights) if weights else None

    def apply(self, x, *args: Any, **kwargs: Any) -> Any:
        if self._weights:
            x = np.multiply(np.array(x), self._weights)
        return np.min(x, *args, **kwargs)


class Aggregator:
    m = {
        "mean": Mean,
        "max": Max,
        "min": Min,
    }

    @staticmethod
    def register(name: str, cls_build_func: Callable):
        Aggregator.m.update(name, cls_build_func)

    @staticmethod
    def get(name: str) -> Union[Callable, None]:
        return Aggregator.m.get(name, None)
