from abc import ABCMeta
from typing import Sequence


class TuneUnit(metaclass=ABCMeta):
    def __init__(self):
        pass


class Grid(TuneUnit):
    def __init__(self, data: Sequence):
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data
