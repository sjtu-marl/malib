import time
import grpc
from threading import Event
from concurrent import futures
from collections import Iterable

from malib.rpc.proto import log_pb2_grpc, log_pb2
from malib.utils import io_wrapper
from malib.utils.convert import utc_to_str, dump_dict
from malib.utils.io_wrapper import BaseIOWrapper, StandardIOWrapper


class LogServicer(log_pb2_grpc.LogRPCServicer):
    def __init__(self, timeout=-1, ioers=None):
        super().__init__()
        self.timeout = timeout
        self.ioers = []
        if isinstance(ioers, Iterable):
            for i in ioers:
                assert isinstance(i, BaseIOWrapper)
                self.ioers.append(i)
        elif ioers is not None:
            assert isinstance(ioers, BaseIOWrapper)
            self.ioers.append(ioers)
        else:
            self.ioers.append(StandardIOWrapper())

        self.alivetime = time.time()

    def Log(self, info, context):
        status = 0
        target = None
        try:
            level = int(info.log_level)
            msg = info.log_info
            st = info.send_time
            self.alivetime = time.time()

            target = {
                "ReceiveTime": time.time(),
                "SendTime": st,
                "Level": level,
                "Content": msg,
            }

        except:
            status = -1
            target = {
                "ReceiveTime": time.time(),
                "SendTime": "N/A",
                "Level": "N/A",
                "Content": "Error",
            }

        for i in self.ioers:
            i.write("LoggerServer: " + dump_dict(target))

        return log_pb2.LogReply(status_code=str(status), send_time=time.time())

    # def stop(self):
    #     for i in self.ioers:
    #         i.write('LoggerServer: Calling server stop')


class LoggerServer:
    def __init__(self, port, io_wrappers=None, grace=5, max_workers=10):
        self.port = port
        self.grace = grace
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        self.io_wrappers = io_wrappers
        log_pb2_grpc.add_LogRPCServicer_to_server(
            LogServicer(ioers=io_wrappers), self.server
        )
        self.server.add_insecure_port(port)

    def start(self):
        self.server.start()

    def stop(self):
        for i in self.io_wrappers:
            i.write("LoggerServer: Calling server stop")
        self.server.stop(grace=self.grace)


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    log_pb2_grpc.add_LogRPCServicer_to_server(LogServicer(), server)
    server.add_insecure_port(port)
    server.start()
    server.wait_for_termination()
