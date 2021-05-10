from .rollout_worker import RolloutWorker
from .sync_rollout_worker import SyncRolloutWorker


def get_rollout_worker(type_name: str) -> type:
    """Return rollout worker type with type name.

    :param str type_name: Rollout typename, choices {async, sync}
    :return: type
    """
    return {"async": RolloutWorker, "sync": SyncRolloutWorker}[type_name]
