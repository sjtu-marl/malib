from typing import Any, Dict

import time
import random
import multiprocessing
import pytest
import threading
import numpy as np

from gym import spaces

from malib.backend.dataset_server.data_loader import DynamicDataset
from malib.backend.dataset_server.utils import send_data, start_server
from malib.backend.dataset_server.feature import BaseFeature


class FakeFeatureHandler(BaseFeature):
    def write(self, data: Dict[str, Any], start: int, end: int):
        print("[FakeFeatureHandler] write data, size:", self._available_size)
        return super().write(data, start, end)

    def get(self, index: int):
        print("[FakeFeatureHandler] get data for index={}".format(index))
        return super().get(index)

    @classmethod
    def gen_instance(cls):
        return cls(
            spaces={
                "a": spaces.Box(-1.0, 1.0, shape=(4,)),
                "b": spaces.Discrete(2),
            },
            block_size=1024,
        )


class TestDynamicDataset:
    def test_grpc_service_write(self):
        grpc_port = 8899
        _spaces = {
            "a": spaces.Box(-1.0, 1.0, shape=(4,)),
            "b": spaces.Discrete(2),
        }
        feature_handler = FakeFeatureHandler(
            spaces=_spaces,
            np_memory={k: np.zeros((1024,) + v.shape) for k, v in _spaces.items()},
        )

        # start server proc
        server_proc = multiprocessing.Process(
            target=start_server,
            args=(
                2,
                1024,
                grpc_port,
                feature_handler,
            ),
        )
        server_proc.start()

        # send data
        for _ in range(10):
            message = send_data(
                feature_handler.generate_timestep(), host="localhost", port=grpc_port
            )
            time.sleep(1)
            print("returned message:", message)

        server_proc.terminate()

    def test_sync_grpc_service_get(self):
        _spaces = {
            "a": spaces.Box(-1.0, 1.0, shape=(4,)),
            "b": spaces.Discrete(2),
        }
        dataset = DynamicDataset(
            grpc_thread_num_workers=2,
            max_message_length=1024,
            feature_handler_cls=FakeFeatureHandler,
            spaces=_spaces,
            np_memory={
                k: np.zeros((1024,) + v.shape, dtype=v.dtype)
                for k, v in _spaces.items()
            },
        )

        # send data
        print("send 10 piece of data, entrypoint=", dataset.entrypoint)
        for _ in range(10):
            message = send_data(
                dataset.feature_handler.generate_batch(batch_size=1),
                entrypoint=dataset.entrypoint,
            )
            time.sleep(1)
            print("returned message:", message)

        # sample data
        assert dataset.readable_block_size == 10, (
            dataset.readable_block_size,
            dataset.feature_handler._available_size,
        )
        for _ in range(10):
            idx = random.randint(0, dataset.readable_block_size - 1)
            data = dataset[idx]
            assert isinstance(data, dict), type(data)
            for k, v in data.items():
                # convert v to numpy
                v = v.cpu().numpy()
                assert dataset.feature_handler.spaces[k].contains(v), (k, v)

        dataset.close()

    def test_async_grpc_service_get(self):
        _spaces = {
            "a": spaces.Box(-1.0, 1.0, shape=(4,)),
            "b": spaces.Discrete(2),
        }
        dataset = DynamicDataset(
            grpc_thread_num_workers=2,
            max_message_length=1024,
            feature_handler_cls=FakeFeatureHandler,
            spaces=_spaces,
            np_memory={
                k: np.zeros((100,) + v.shape, dtype=v.dtype) for k, v in _spaces.items()
            },
        )

        def start_send(batch, entrypoint):
            print("send 10 piece of data, entrypoint=", entrypoint)
            for data in batch:
                message = send_data(
                    data,
                    entrypoint=entrypoint,
                )
                time.sleep(1)
                print("returned message:", message)

        batch = [
            dataset.feature_handler.generate_batch(batch_size=1) for _ in range(10)
        ]
        entrypoint = dataset.entrypoint
        send_proc = threading.Thread(target=start_send, args=(batch, entrypoint))

        send_proc.start()

        def start_get():
            while dataset.readable_block_size == 0:
                time.sleep(0.1)

            for _ in range(10):
                idx = random.randint(0, dataset.readable_block_size)
                data = dataset[idx]
                assert isinstance(data, dict), type(data)
                for k, v in data.items():
                    # convert v to numpy
                    v = v.cpu().numpy()
                    assert dataset.feature_handler.spaces[k].contains(v), (k, v)

        get_proc = threading.Thread(target=start_get)
        get_proc.start()

        send_proc.join()
        get_proc.join()
        dataset.close()
