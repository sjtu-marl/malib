import threading
import time
from types import LambdaType

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


def run(
    task_mode: str,
    env_description: Dict[str, Any],
    training: Dict[str, Any],
    algorithms: Dict[str, Any],
    rollout_worker: Dict[str, Any],
    group: str = "experiment",
    name: str = str(time.time()),
    agent_mapping_func: LambdaType = lambda agent: agent,
    evaluation: Dict[str, Any] = dict(),
    global_evaluator: Any = None,
    dataset_config: Dict[str, Any] = dict(),
    parameter_server: Dict[str, Any] = dict(),
):
    """Launch learning task.

    :param group: Naming the experiment group
    :type group: str
    :param name: Specifying the experiment name
    :type name: str
    :param task_mode: Task mode, could be `marl` or `gt`
    :type task_mode: str
    :param env_description: Environment description
    :type env_description: Dict[str, Any]
    :param training: Training configuration
    :type training: Dict[str, Any]
    :param algorithms: Algorithm configuration
    :type algorithms: Dict[str, Any]
    :param rollout_worker: Rollout configuration for worker initialization
    :type rollout_worker: Dict[str, Any]
    :param agent_mapping_func: Agent mapping function, will determine which agent will be mapped into one learner, defaults to lambdaagent:agent
    :type agent_mapping_func: LambdaType, optional
    :param evaluation: Evaluation configuration, defaults to None
    :type evaluation: Dict[str, Any], optional
    :param global_evaluator: Specifying the global evaluator configuration, defaults to None
    :type global_evaluator: Any, optional
    :param dataset_config: Dataset configuration, defaults to None
    :type dataset_config: Dict[str, Any], optional
    :param parameter_server: Parameter server configuration, defaults to None
    :type parameter_server: Dict[str, Any], optional
    """

    config = locals()
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
        except ConnectionError:
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
        ).remote(global_configs["dataset"])
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
