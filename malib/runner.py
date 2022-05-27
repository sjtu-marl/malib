import ray

from malib import settings
from malib.utils.logging import Logger
from malib.scenarios import marl_scenario, psro_scenario
from malib.scenarios.scenario import Scenario
from malib.backend.offline_dataset_server import OfflineDataset
from malib.backend.parameter_server import ParameterServer


def run(scenario: Scenario):
    try:
        start_ray_info = ray.init(address="auto")
    except ConnectionError:
        Logger.warning("No active cluster deteced, will create a local ray instance.")
        start_ray_info = ray.init()

    Logger.info("Ray lauched: {}".format(start_ray_info))
    Logger.info("Ray cluster resources info: {}".format(ray.cluster_resources()))

    offline_dataset = OfflineDataset.options(
        name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
    ).remote(scenario.dataset_config)
    parameter_server = ParameterServer.options(
        name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
    ).remote(**scenario.parameter_server_config)

    _ = ray.get([offline_dataset.start.remote(), parameter_server.start.remote()])

    if isinstance(scenario, marl_scenario.MARLScenario):
        marl_scenario.execution_plan(scenario)
    elif isinstance(scenario, psro_scenario.PSROScenario):
        psro_scenario.execution_plan(scenario)
    else:
        raise TypeError("Unexpected scenario type: {}".format(scenario))
