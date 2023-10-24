from typing import Type, Any

import threading
import grpc

from concurrent import futures
from torch.utils.data import DataLoader, Dataset

from malib.utils.general import find_free_port

from .service import DatasetServer
from . import data_pb2_grpc
from .feature import BaseFeature


class EmptyError(Exception):
    pass


class DynamicDataset(Dataset):
    def __init__(
        self,
        grpc_thread_num_workers: int,
        max_message_length: int,
        feature_handler_caller: Type,
    ) -> None:
        super().__init__()

        # start a service as thread
        self.feature_handler: BaseFeature = feature_handler_caller()
        self.server = self._start_servicer(
            grpc_thread_num_workers,
            max_message_length,
            find_free_port(),
        )

    def _start_servicer(
        self, max_workers: int, max_message_length: int, grpc_port: int
    ):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", max_message_length),
                ("grpc.max_receive_message_length", max_message_length),
            ],
        )
        servicer = DatasetServer(self.feature_handler)
        data_pb2_grpc.add_SendDataServicer_to_server(servicer, server)

        server.add_insecure_port(f"[::]:{grpc_port}")
        server.start()

        return server

    def __len__(self):
        return self.feature_handler_caller.block_size

    def __getitem__(self, index) -> Any:
        if index >= len(self):
            raise IndexError

        if len(self.feature_handler) == 0:
            raise EmptyError(f"No available data for sampling")

        return self.feature_handler.safe_get(index)

    def close(self):
        self.server.stop()
