import sys

from .base_io_wrapper import BaseIOWrapper


class StandardIOWrapper(BaseIOWrapper):
    def __init__(self, in_stream=sys.stdin, out_stream=sys.stdout):
        self.in_stream = in_stream
        self.out_stream = out_stream

    def write(self, object, serialzer=None):
        content = ""
        if isinstance(object, str):
            content = object
        elif serialzer:
            content = serialzer(object)
        elif hasattr(object, "__serialize__"):
            content = object.__serialize__()
        else:
            content = str(object)
        self.out_stream.write(content)
        self.out_stream.flush()

    def read(self):
        return self.in_stream.read()
