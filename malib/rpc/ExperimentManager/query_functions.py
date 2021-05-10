import abc
import numpy as np
from malib.utils.typing import Tuple


class QueryFunction(abc.ABC):
    def __init__(
        self,
        name: str,
        tag: str,
        sub_tags: Tuple[str] = None,
        op_nums: Tuple[int] = None,
    ):
        self._name = name
        self._tag = tag
        self._sub_tags = sub_tags

        if self._sub_tags is None:
            if op_nums is None:
                self._op_nums = tuple([0])
            else:
                self._op_nums = tuple(op_nums[:1])
        else:
            try:
                sub_tags = tuple(sub_tags)
                op_nums = tuple(op_nums)
            except Exception:
                raise RuntimeError("Except tuple or iterable input of sub-tags")

            if len(sub_tags) <= len(op_nums):
                self._op_nums = op_nums[: len(sub_tags)]
            else:
                op_nums = list(op_nums)
                op_nums.extend([op_nums[-1]] * len(sub_tags))
                op_nums = tuple(op_nums)
            self._op_nums = op_nums

    @abc.abstractmethod
    def apply(self, x):
        pass

    def name(self):
        return self._name

    def _generate_query(self):
        pass


class Query(QueryFunction):
    def __init__(self, tag: str, sub_tags: Tuple[str], op_nums: Tuple[int]):
        super().__init__(name="Query", tag=tag, sub_tags=sub_tags, op_nums=op_nums)

    def apply(self, x):
        return x


class Generic(QueryFunction):
    def __init__(
        self,
        name,
        func,
        tag: str,
        sub_tags: Tuple[str],
        op_nums: Tuple[int],
        *args,
        **kwargs
    ):
        super(Generic, self).__init__(
            name=name, tag=tag, sub_tags=sub_tags, op_nums=op_nums
        )
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def apply(self, x):
        return self._func(x, *self._args, **self._kwargs)


class Mean(QueryFunction):
    def __init__(
        self, tag: str, sub_tags: Tuple[str], op_nums: Tuple[int], *args, **kwargs
    ):
        super().__init__(name="Mean", tag=tag, sub_tags=sub_tags, op_nums=op_nums)
        self._func = np.mean
        self._args = args
        self._kwargs = kwargs

    def apply(self, x):
        return self._func(x, *self._args, **self._kwargs)


class Max(QueryFunction):
    def __init__(
        self, tag: str, sub_tags: Tuple[str], op_nums: Tuple[int], *args, **kwargs
    ):
        super().__init__(name="Max", tag=tag, sub_tags=sub_tags, op_nums=op_nums)
        self._func = np.max
        self._args = args
        self._kwargs = kwargs

    def apply(self, x):
        return self._func(x, *self._args, **self._kwargs)


class Min(QueryFunction):
    def __init__(
        self, tag: str, sub_tags: Tuple[str], op_nums: Tuple[int], *args, **kwargs
    ):
        super().__init__(name="Min", tag=tag, sub_tags=sub_tags, op_nums=op_nums)
        self._func = np.min
        self._args = args
        self._kwargs = kwargs

    def apply(self, x):
        return self._func(x, *self._args, **self._kwargs)
