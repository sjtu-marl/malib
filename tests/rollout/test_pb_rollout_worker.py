# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

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

from typing import Dict, Any

import pytest
import threading
import time
import ray

from pytest_mock import MockerFixture
from gym import spaces

from malib.runner import start_servers
from malib.mocker.mocker_utils import FakeInferenceClient, FakeInferenceServer


def gen_rollout_config(inference_server_type: str):
    return {
        "fragment_length": 100,
        "max_step": 10,
        "num_eval_episodes": 2,
        "num_threads": 1,
        "num_env_per_thread": 1,
        "num_eval_threads": 1,
        "use_subproc_env": False,
        "batch_mode": "timestep",
        "postprocessor_types": None,
        "eval_interval": 1,
        "inference_server": inference_server_type,
    }


def create_rollout_worker(
    mocker: MockerFixture, env_desc: Dict[str, Any], rollout_config: Dict[str, Any]
):
    mocker.patch(
        "malib.rollout.rolloutworker.RayInferenceClient", new=FakeInferenceClient
    )
    mocker.patch(
        "malib.rollout.rolloutworker.RayInferenceServer", new=FakeInferenceServer
    )
    from malib.rollout.pb_rolloutworker import PBRolloutWorker

    worker = PBRolloutWorker(
        experiment_tag="test_rollout_worker",
        env_desc=env_desc,
        agent_mapping_func=lambda agent: agent,
        rollout_config=rollout_config,
        log_dir="./logs",
    )
    return worker


@pytest.mark.parametrize("n_player", [1, 2])
@pytest.mark.parametrize("inference_server_type", ["local", "ray"])
class TestRolloutWorker:
    def test_rollout(
        self, mocker: MockerFixture, n_player: int, inference_server_type: str
    ):
        if not ray.is_initialized():
            ray.init()

        parameter_server, dataset_server = start_servers()

        agents = [f"player_{i}" for i in range(n_player)]

        worker = create_rollout_worker(
            mocker,
            env_desc={
                "possible_agents": agents,
                "observation_spaces": {
                    agent: spaces.Box(-1, 1.0, shape=(2,)) for agent in agents
                },
                "action_spaces": {
                    agent: spaces.Box(-1, 1, shape=(2,)) for agent in agents
                },
            },
            rollout_config=gen_rollout_config(inference_server_type),
        )

        data_entrypoints = {agent: agent for agent in agents}
        results = worker.rollout(
            None,
            {"max_iteration": 2},
            data_entrypoints,
            None,
        )
        print("rollout results:", results)

        ray.kill(parameter_server)
        ray.kill(dataset_server)
        ray.shutdown()

    def test_simulation(
        self, mocker: MockerFixture, n_player: int, inference_server_type: str
    ):
        if not ray.is_initialized():
            ray.init()

        parameter_server, dataset_server = start_servers()

        agents = [f"player_{i}" for i in range(n_player)]

        worker = create_rollout_worker(
            mocker,
            env_desc={
                "possible_agents": agents,
                "observation_spaces": {
                    agent: spaces.Box(-1, 1.0, shape=(2,)) for agent in agents
                },
                "action_spaces": {
                    agent: spaces.Box(-1, 1, shape=(2,)) for agent in agents
                },
            },
            rollout_config=gen_rollout_config(inference_server_type),
        )

        results = worker.simulate({})

        ray.kill(parameter_server)
        ray.kill(dataset_server)
        ray.shutdown()
