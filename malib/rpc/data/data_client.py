import grpc

import sys

sys.path.append("..")
from ..proto import data_pb2_grpc, data_pb2


def send(server_port, **kargs):
    with grpc.insecure_channel(server_port) as channel:
        stub = data_pb2_grpc.DataRPCStub(channel)
        pr = data_pb2.PullRequest(
            type=kargs["tid"], schema_id=kargs["sid"], instance_id=kargs["iid"]
        )
        data = stub.Pull(pr)
