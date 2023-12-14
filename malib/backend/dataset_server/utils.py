from typing import Any, Union
from concurrent import futures

import sys
import os
import pickle
import grpc

sys.path.append(os.path.dirname(__file__))

from .service import DatasetServer
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


def service_wrapper(max_workers: int, max_message_length: int, grpc_port: int):
    def func(feature_handler):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", max_message_length),
                ("grpc.max_receive_message_length", max_message_length),
            ],
        )
        servicer = DatasetServer(feature_handler)
        data_pb2_grpc.add_SendDataServicer_to_server(servicer, server)

        server.add_insecure_port(f"[::]:{grpc_port}")
        return server

    return func


def start_server(
    max_workers: int, max_message_length: int, grpc_port: int, feature_handler
):
    server = service_wrapper(
        max_workers=max_workers,
        max_message_length=max_message_length,
        grpc_port=grpc_port,
    )(feature_handler)
    server.start()
    server.wait_for_termination()
