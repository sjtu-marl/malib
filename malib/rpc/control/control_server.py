import time
import grpc
from concurrent import futures

import sys

sys.path.append("..")
from ..proto import control_pb2_grpc, control_pb2


class ControlServicer(control_pb2_grpc.ControlRPCServicer):
    def __init__(self, regis_table=None):
        super().__init__()
        self._regis_table = regis_table

    def HeatBeat(self, request, context):
        if self._regis_table:
            res = self.regis_table.update(
                request.node_type,
                request.node_id,
                request.node_status,
                request.send_time,
            )
            return control_pb2.BeatReply(
                target_code=res[0], action_code=res[1], send_time=time.time()
            )
        else:
            print(request.node_type)
            print(request.node_id)
            print(request.node_status)
            print(request.send_time)
            return control_pb2.BeatReply(
                target_code="0", action_code="0", send_time=time.time()
            )


def serve(port, regis_table=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    control_pb2_grpc.add_ControlRPCServicer_to_server(
        ControlServicer(regis_table), server
    )
    server.add_insecure_port(port)
    server.start()
    server.wait_for_termination()
