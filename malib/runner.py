import time
import ray

from malib import settings
from malib.utils.logging import Logger
from malib.scenarios import marl_scenario, psro_scenario
from malib.scenarios.scenario import Scenario
from malib.backend.offline_dataset_server import OfflineDataset
from malib.backend.parameter_server import ParameterServer


def start_servers():
    try:
        offline_dataset_server = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR
        ).remote(table_capacity=100)
    except ValueError:
        Logger.warning("detected existing offline dataset server")
        offline_dataset_server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)

    try:
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR
        ).remote()
    except ValueError:
        Logger.warning("detected exisitng parameter server")
        parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)

    ray.get([parameter_server.start.remote(), offline_dataset_server.start.remote()])
    return parameter_server, offline_dataset_server


def run(scenario: Scenario):
    try:
        start_ray_info = ray.init(address="auto")
    except ConnectionError:
        Logger.warning("No active cluster deteced, will create a local ray instance.")
        start_ray_info = ray.init()

    Logger.info("Ray lauched: {}".format(start_ray_info))
    Logger.info("Ray cluster resources info: {}".format(ray.cluster_resources()))

    _ = start_servers()

    experiment_tag = f"malib-{scenario.name}-{time.time()}"

    if isinstance(scenario, marl_scenario.MARLScenario):
        marl_scenario.execution_plan(experiment_tag, scenario)
    elif isinstance(scenario, psro_scenario.PSROScenario):
        psro_scenario.execution_plan(scenario)
    else:
        raise TypeError("Unexpected scenario type: {}".format(scenario))
