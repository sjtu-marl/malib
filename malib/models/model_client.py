from typing import Dict, Any
from concurrent import futures

import threading
import torch
import ray

from readerwriterlock import rwlock
from torch import nn

from malib.models.config import ModelConfig


def load_state_dict(client, timeout=10):
    if isinstance(client, ray.ObjectRef):
        return ray.get(client.get_state_dict.remote(), timeout=10)
    else:
        raise NotImplementedError


class ModelClient:
    def __init__(self, entry_point: str, model_config: ModelConfig):
        """Construct a model client for mantaining a model instance and its update.

        Args:
            entry_point (str): Entry point for model update.
            model_cls (nn.Module): Model class for constructing model instance.
            model_args (Dict[str, Any]): Arguments for constructing model instance.

        Raises:
            NotImplementedError: Unsupported cluster type.
        """

        cluster_type, name_or_address = entry_point.split(":")

        if "ray" in cluster_type:
            self.client = ray.get_actor(name_or_address)
        else:
            raise NotImplementedError

        self.cluster_type = cluster_type
        self.server_address = name_or_address
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=10)

        self.event = threading.Event()
        self.thread_pool.submit(self._model_update, self.event)
        self.model: nn.Module = model_config.model_cls(**model_config.model_args).cpu()
        self.model.share_memory()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with torch.inference_mode():
            return self.model(*args, **kwds)

    def actor(self, *args, **kwargs):
        return self.model.actor(*args, **kwargs)

    def critic(self, *args, **kwargs):
        return self.model.critic(*args, **kwargs)

    def _model_update(self, event: threading.Event):
        while not event.is_set():
            # TODO(ming): update model from remote server
            try:
                state_dict = load_state_dict(self.client)

                event.wait(0.5)
            except TimeoutError:
                # TODO(ming): count or reconnect
                event.wait(1)
            except RuntimeError:
                pass
            except KeyboardInterrupt:
                break

    def shutdown(self):
        self.event.set()
        self.thread_pool.shutdown()
