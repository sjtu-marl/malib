from typing import Any
from readerwriterlock import rwlock


class BaseFeature:
    def __init__(self) -> None:
        self.rw_lock = rwlock.RWLockFair()
        self._readable_index = []
        self._writable_index = []

    @property
    def block_size(self) -> int:
        raise NotImplementedError

    def __len__(self):
        return len(self._readable_index)

    def _get(self, index: int):
        raise NotImplementedError

    def safe_get(self, index: int):
        with self.rw_lock.gen_rlock():
            return self._get(index)

    def _write(self, data: Any):
        raise NotImplementedError

    def safe_put(self, data: Any):
        with self.rw_lock.gen_wlock():
            self._write(data)
