from typing import Type, Any

import socket
import threading
import grpc

from torch.utils.data import Dataset

from malib.utils.general import find_free_port
from malib.backend.dataset_server.utils import service_wrapper

from .feature import BaseFeature


class EmptyError(Exception):
    pass


class DynamicDataset(Dataset):
    def __init__(
        self,
        grpc_thread_num_workers: int,
        max_message_length: int,
        feature_handler: BaseFeature = None,
        feature_handler_cls: Type[BaseFeature] = None,
        **feature_handler_kwargs,
    ) -> None:
        super().__init__()

        # start a service as thread
        self.feature_handler: BaseFeature = feature_handler or feature_handler_cls(
            **feature_handler_kwargs
        )
        self.grpc_thread_num_workers = grpc_thread_num_workers
        self.max_message_length = max_message_length

    def start_server(self):
        self.server_port = find_free_port()
        self.server = service_wrapper(
            self.grpc_thread_num_workers,
            self.max_message_length,
            self.server_port,
        )(self.feature_handler)
        self.server.start()
        self.host = socket.gethostbyname(socket.gethostname())

    @property
    def entrypoint(self) -> str:
        return f"{self.host}:{self.server_port}"

    @property
    def readable_block_size(self) -> str:
        return len(self.feature_handler)

    def __len__(self):
        return self.feature_handler.block_size

    def __getitem__(self, index) -> Any:
        if index >= len(self):
            raise IndexError

        if len(self.feature_handler) == 0:
            raise EmptyError(f"No available data for sampling")

        return self.feature_handler.safe_get(index)

    def close(self):
        self.server.wait_for_termination(3)
