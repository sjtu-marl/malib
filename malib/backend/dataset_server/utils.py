from typing import Any, Union

import pickle
import grpc

from . import data_pb2
from . import data_pb2_grpc


def send_data(host: str, port: int, data: Any):
    if not isinstance(data, bytes):
        data = pickle.dumps(data)

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = data_pb2_grpc.SendDataStub(channel)
        reply = stub.Collect(data_pb2.Data(data=data))

    return reply.message
