import abc
from malib.utils.typing import Any


class BaseClient(abc.ABC):
    def info(self, level: str, message: str, nid: str):
        pass

    @abc.abstractmethod
    def create_table(self, primary: str, secondary: str, nid: str):
        pass

    @abc.abstractmethod
    def send_scalar(self, key: int, tag: str, nid: str, content: Any):
        pass

    @abc.abstractmethod
    def send_image(self, key, tag, image, serial):
        pass

    @abc.abstractmethod
    def send_figure(self, key, tag, nid, figure):
        pass

    @abc.abstractmethod
    def send_obj(self, key, tag, nid, obj, serial):
        pass

    @abc.abstractmethod
    def send_binary_tensor(self, key, tag, nid, tensor):
        pass
