import copy
import pprint
import threading
import time
from typing import Counter, Dict, Any, List

import ray

from malib import settings
from malib.utils import logger
from malib.utils.logger import Logger, get_logger, Log, start
from malib.utils.configs.formatter import DefaultConfigFormatter
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

    infos = DefaultConfigFormatter.parse(global_configs)

    exp_cfg = logger.start(
        group=global_configs.get("group", "experiment"),
        name=global_configs.get("name", "case") + f"_{time.time()}",
    )

    try:
        from malib.backend.coordinator.task import CoordinatorServer
        from malib.backend.datapool.offline_dataset_server import OfflineDataset
        from malib.backend.datapool.parameter_server import ParameterServer

        Logger.info(
            "Pre launch checking for Coordinator server ... {}".format(
                getattr(CoordinatorServer, "_request_simulation", None)
            )
        )
        pprint.pprint(f"Logged experiment information:{infos}", indent=2)

        ray.init(local_mode=False)

        offline_dataset = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
        ).remote(global_configs["dataset_config"], exp_cfg)
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(exp_cfg=exp_cfg, **global_configs["parameter_server"])
        coordinator_server = CoordinatorServer.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote(exp_cfg=exp_cfg, **global_configs)

        _ = ray.get(
            coordinator_server.start.remote(
                use_init_policy_pool=config.get("use_init_policy_pool", True)
            )
        )

        performance_logger = get_logger(
            name="performance",
            expr_group=exp_cfg["expr_group"],
            expr_name=exp_cfg["expr_name"],
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            info=infos,
        )

        with Log.timer(
            log=settings.PROFILING,
            logger=performance_logger,
        ):
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
        print(
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
