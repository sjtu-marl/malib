import time
import grpc

from malib.rpc.proto import log_pb2_grpc, log_pb2
from malib.utils.convert import utc_to_str


class LoggerClient:
    def __init__(self, server_addr):
        self.server_addr = server_addr

    def send(self, msg, level=1):
        with grpc.insecure_channel(self.server_addr) as channel:
            stub = log_pb2_grpc.LogRPCStub(channel)
            info = log_pb2.LogInfo(
                log_level=str(level), log_info=msg, send_time=time.time()
            )
            response = stub.Log(info)
            return response.status_code, response.send_time


def run(port):
    channel = grpc.insecure_channel(port)
    stub = log_pb2_grpc.LogRPCStub(channel)

    while True:
        info = log_pb2.LogInfo(log_level=str(1), log_info="Test", send_time=time.time())
        response = stub.Log(info)
        print(
            "Log client received reply: [status: {}, time: {}]".format(
                response.status_code, utc_to_str(response.send_time)
            )
        )
        time.sleep(1)
