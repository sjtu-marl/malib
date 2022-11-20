# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import ray

from malib import settings
from malib.utils.logging import Logger
from malib.scenarios import marl_scenario, psro_scenario
from malib.scenarios.scenario import Scenario
from malib.backend.offline_dataset_server import OfflineDataset
from malib.backend.parameter_server import ParameterServer


def start_servers(data_table_capacity: int = 100000):
    try:
        offline_dataset_server = (
            OfflineDataset.as_remote(num_cpus=0)
            .options(name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=100)
            .remote(table_capacity=data_table_capacity)
        )
        ray.get(offline_dataset_server.start.remote())
    except ValueError:
        Logger.warning("detected existing offline dataset server")
        offline_dataset_server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)

    try:
        parameter_server = (
            ParameterServer.as_remote(num_cpus=1)
            .options(name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=100)
            .remote()
        )
        ray.get(parameter_server.start.remote())
    except ValueError:
        Logger.warning("detected exisitng parameter server")
        parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)

    return parameter_server, offline_dataset_server


def run(scenario: Scenario, cluster_address: str = "auto"):
    """Load scenario to the execution plan and lauch a cluster. The instance will search an active \
        cluster by default. Users can also determine the specified cluster with given `cluster_address`.

    Args:
        scenario (Scenario): Scenario instance.
        cluster_address (str, optional): Ray cluster address. Defaults to "auto", which means the \
            training instance will search an active cluster.

    Raises:
        TypeError: Unexpected scenario type.
    """

    try:
        start_ray_info = ray.init(address="auto", dashboard_port=8265)
    except ConnectionError:
        Logger.warning("No active cluster deteced, will create a local ray instance.")
        start_ray_info = ray.init()

    try:
        Logger.info("Ray lauched: {}".format(start_ray_info))
        Logger.info("Ray cluster resources info: {}".format(ray.cluster_resources()))

        parameter_server, offline_dataset_server = start_servers()
        scenario.parameter_server = parameter_server
        scenario.offline_dataset_server = offline_dataset_server

        experiment_tag = f"malib-{scenario.name}-{time.time()}"

        if isinstance(scenario, psro_scenario.PSROScenario):
            psro_scenario.execution_plan(experiment_tag, scenario)
        elif isinstance(scenario, marl_scenario.MARLScenario):
            marl_scenario.execution_plan(experiment_tag, scenario)
        else:
            raise TypeError("Unexpected scenario type: {}".format(scenario))
    except KeyboardInterrupt:
        ray.shutdown()
    except TypeError as e:
        ray.shutdown()
        raise e
    except Exception as e:
        raise e
