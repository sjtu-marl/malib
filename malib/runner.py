import copy
import threading
import time
import traceback
from typing import Dict, Any, List

import ray

from malib import settings
from malib.utils.logger import get_logger, Log
from malib.utils.configs.formatter import DefaultConfigFormatter
from malib.rpc.ExperimentManager.ExperimentServer import start_logging_server
from malib.utils.logger import get_logger

WAIT_FOR_READY_THRESHOLD = 10
logger_server = None


def update_configs(update_dict, ori_dict=None):
    """ Update global configs with a given dict """

    ori_configs = (
        copy.copy(ori_dict)
        if ori_dict is not None
        else copy.copy(settings.DEFAULT_CONFIG)
    )

    for k, v in update_dict.items():
        # assert k in ori_configs, f"Illegal key: {k}, {list(ori_configs.keys())}"
        if isinstance(v, dict):
            ph = ori_configs[k] if isinstance(ori_configs.get(k), dict) else {}
            ori_configs[k] = update_configs(v, ph)
        else:
            ori_configs[k] = copy.copy(v)
    return ori_configs


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


def start_logger_server(exp_cfg):
    global logger_server
    logger_server = start_logging_server(
        port="localhost:12333", logdir=settings.LOG_DIR
    )
    logger_server.start()


def terminate_logger():
    global logger_server
    logger_server.terminate()


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
    print(f"Logged experiment information:{infos}")

    exp_cfg = {
        "expr_group": global_configs.get("group", "experiment"),
        "expr_name": f"{global_configs.get('name', 'case')}_{time.time()}",
    }

    ray.init(local_mode=False)
    try:
        # def run_coordinator_server(coordinator_server_configs):
        if settings.USE_REMOTE_LOGGER and not settings.USE_MONGO_LOGGER:
            start_logger_server(exp_cfg)

        # wait for the logging server to be ready
        _wait_for_ready_start_time = time.time()
        while True:
            try:
                # logger = get_logger(port="localhost:12333", remote=True)
                logger = get_logger(
                    name="runner",
                    expr_group=exp_cfg["expr_group"],
                    expr_name=exp_cfg["expr_name"],
                    remote=settings.USE_REMOTE_LOGGER,
                    mongo=settings.USE_MONGO_LOGGER,
                    info=infos,
                )
                logger.info("Wait for server ready")
                break
            except Exception as e:
                if time.time() - _wait_for_ready_start_time > WAIT_FOR_READY_THRESHOLD:
                    raise RuntimeError(
                        "Wait time exceed threshold, "
                        "task cancelled, "
                        "cannot connect to logging server, "
                        "please check the network availability!"
                    )
                time.sleep(1)

        from malib.backend.coordinator.server import CoordinatorServer
        from malib.backend.datapool.offline_dataset_server import OfflineDataset
        from malib.backend.datapool.parameter_server import ParameterServer

        offline_dataset = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
        ).remote(global_configs["dataset_config"], exp_cfg)
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(exp_cfg=exp_cfg, **global_configs["parameter_server"])

        coordinator_server = CoordinatorServer.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote(exp_cfg=exp_cfg, **global_configs)
        _ = ray.get(coordinator_server.start.remote())

        with Log.timer(log=settings.PROFILING, logger=logger):
            while True:
                terminate = ray.get(coordinator_server.is_terminate.remote())
                if terminate:
                    print("ALL task done")
                    break
                else:
                    time.sleep(1)
        _terminate(
            [
                {"func": ray.shutdown, "args": tuple()},
                {"func": logger_server.terminate, "args": tuple()},
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
                {"func": logger_server.terminate, "args": tuple()},
                {"func": offline_dataset.shutdown.remote, "args": ()},
                {"func": parameter_server.shutdown.remote, "args": ()},
            ],
            waiting=False,
        )
        # sys.exit(0)
