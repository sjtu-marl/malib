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

import time

from collections import deque


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


class AvgTime:
    def __init__(self, num_values_to_avg):
        self.values = deque([], maxlen=num_values_to_avg)

    def tofloat(self):
        return sum(self.values) / max(1, len(self.values))

    def __str__(self):
        avg_time = sum(self.values) / max(1, len(self.values))
        return f"{avg_time:.4f}"


class TimingContext:
    def __init__(self, timer, key, additive=False, average=None):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None

    def __enter__(self):
        self._time_enter = time.time()

    def __exit__(self, type_, value, traceback):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = AvgTime(num_values_to_avg=self._average)
            else:
                self._timer[self._key] = 0

        time_passed = max(time.time() - self._time_enter, 1e-6)

        if self._additive:
            self._timer[self._key] += time_passed
        elif self._average is not None:
            self._timer[self._key].values.append(time_passed)
        else:
            self._timer[self._key] = time_passed


class Timing(AttrDict):
    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key: str):
        """Add time additively.
        Args:
            key (str): Timer key.
        Returns:
            TimingContext: A `TimingContext` instance in additive mode..
        """

        return TimingContext(self, key, additive=True)

    def time_avg(self, key, average=10):
        return TimingContext(self, key, average=average)

    def todict(self):
        res = {}
        for k, v in self.items():
            if isinstance(v, float):
                res[k] = v
            else:
                res[k] = v.tofloat()
        return res

    def __str__(self):
        s = ""
        i = 0
        for key, value in self.items():
            str_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            s += f"{key}: {str_value}"
            if i < len(self) - 1:
                s += ", "
            i += 1
        return s
