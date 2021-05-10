import time
import grpc

import sys

sys.path.append("..")
from ..proto import control_pb2_grpc, control_pb2


def run(server_port):
    channel = grpc.insecure_channel(server_port)
    stub = control_pb2_grpc.ControlRPCStub(channel)

    while True:
        sig = control_pb2.BeatSignal(
            node_type="0", node_id="0", node_status="normal", send_time=time.time()
        )
        response = stub.HeatBeat(sig)
        print(
            "Control client received reply: [target: {}, action: {}, time: {}]".format(
                response.target_code, response.action_code, response.send_time
            )
        )
        time.sleep(1)
