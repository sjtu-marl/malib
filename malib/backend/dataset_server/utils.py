from typing import Any, Union

import pickle
import grpc
import sys
import os

sys.path.append(os.path.dirname(__file__))

from . import data_pb2
from . import data_pb2_grpc


def send_data(data: Any, host: str = None, port: int = None, entrypoint: str = None):
    if not isinstance(data, bytes):
        data = pickle.dumps(data)

    if host is not None:
        with grpc.insecure_channel(f"{host}:{port}") as channel:
            stub = data_pb2_grpc.SendDataStub(channel)
            reply = stub.Collect(data_pb2.Data(data=data))
    else:
        with grpc.insecure_channel(entrypoint) as channel:
            stub = data_pb2_grpc.SendDataStub(channel)
            reply = stub.Collect(data_pb2.Data(data=data))

    return reply.message
