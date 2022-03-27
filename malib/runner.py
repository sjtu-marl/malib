import threading
import time

import ray

from malib import settings
from malib.utils.typing import Dict, Any, List
from malib.utils import logger
from malib.utils.logger import Logger
from malib.utils.general import update_configs


def _terminate(recycle_funcs: List[Dict[str, Any]], waiting: bool = True):
    background_recycle_threads = []
    for call in recycle_funcs:
        background_recycle_threads.append(
            threading.Thread(target=call["func"], args=call["args"])
        )
    for thread in background_recycle_threads:
        thread.start()
    if waiting:
        for thread in background_recycle_threads:
            thread.join()
        print("Background recycling thread ended.")


def run(**kwargs):
    config = locals()["kwargs"]
    global_configs = update_configs(config)
    if global_configs["training"]["interface"].get("worker_config") is None:
        global_configs["training"]["interface"]["worker_config"] = {
            "num_cpus": None,
            "num_gpus": None,
            "memory": None,
            "object_store_memory": None,
            "resources": None,
        }

    try:
        from malib.backend.coordinator.task import CoordinatorServer
        from malib.backend.datapool.offline_dataset_server import OfflineDataset
        from malib.backend.datapool.parameter_server import ParameterServer

        try:
            start_ray_info = ray.init(address="auto")
        except OSError:
            Logger.warning(
                "No active cluster deteced, will create a local ray instance."
            )
            start_ray_info = ray.init()

        Logger.info("Ray lauched: {}".format(start_ray_info))
        Logger.info("Ray cluster resources info: {}".format(ray.cluster_resources()))
        exp_cfg = logger.start(
            group=global_configs.get("group", "experiment"),
            name=global_configs.get("name", "case") + f"_{time.time()}",
            host=start_ray_info["node_ip_address"],
        )

        CoordinatorServer = CoordinatorServer.as_remote()
        ParameterServer = ParameterServer.as_remote()

        offline_dataset = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
        ).remote(global_configs["dataset_config"], exp_cfg)
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(**global_configs["parameter_server"])
        coordinator_server = CoordinatorServer.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote(exp_cfg=exp_cfg, **global_configs)

        _ = ray.get(coordinator_server.start.remote())

        while True:
            terminate = ray.get(coordinator_server.is_terminate.remote())
            if terminate:
                print("ALL task done")
                break
            else:
                time.sleep(1)

        tasks = [offline_dataset.shutdown.remote(), parameter_server.shutdown.remote()]
        while len(tasks) > 0:
            dones, tasks = ray.wait(tasks)

        print("Offline Dataset/ Parameter servering closed")
        _terminate(
            [
                {"func": ray.shutdown, "args": tuple()},
                {"func": logger.terminate, "args": tuple()},
            ],
            waiting=True,
        )

    except KeyboardInterrupt as e:
        Logger.info(
            "Detected KeyboardInterrupt event, start background resources recycling threads ..."
        )
        _terminate(
            [
                {"func": ray.shutdown, "args": tuple()},
                {"func": logger.terminate, "args": tuple()},
                {"func": offline_dataset.shutdown.remote, "args": ()},
                {"func": parameter_server.shutdown.remote, "args": ()},
            ],
            waiting=False,
        )
        # sys.exit(0)
