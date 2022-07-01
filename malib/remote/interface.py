import threading
import ray


class RemoteInterface:
    def __init__(self) -> None:
        self.running = False

    def set_running(self, value):
        self.running = value

    def is_running(self):
        return self.running

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization"""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def stop_pending_tasks(self):
        """External object can call this method to stop all pending tasks."""

        self.set_running(False)
