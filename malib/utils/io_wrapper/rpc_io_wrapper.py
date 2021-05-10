from .base_io_wrapper import BaseIOWrapper


class RpcIOWrapper(BaseIOWrapper):
    def __init__(self, in_stream=None, out_stream=None):
        raise NotImplementedError
