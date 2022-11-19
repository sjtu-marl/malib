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
import ray

from gym import spaces
from pytest_mock import MockerFixture

from malib.common.strategy_spec import StrategySpec
from malib.rollout.manager import RolloutWorkerManager
from malib.mocker.mocker_utils import FakeRolloutWorker


def create_manager(
    mocker: MockerFixture,
    stopping_conditions: Dict[str, Any],
    rollout_config: Dict[str, Any],
    env_desc: Dict[str, Any],
):
    mocker.patch("malib.rollout.manager.PBRolloutWorker", new=FakeRolloutWorker)
    manager = RolloutWorkerManager(
        experiment_tag="test_rollout_manager",
        stopping_conditions=stopping_conditions,
        num_worker=1,
        agent_mapping_func=lambda agent: agent,
        rollout_config=rollout_config,
        env_desc=env_desc,
        log_dir="./logs",
    )
    return manager


@pytest.mark.parametrize("n_players", [1, 2])
@pytest.mark.parametrize("inference_server_type", ["local", "ray"])
class TestRolloutManager:
    def test_rollout_task_send(
        self, mocker: MockerFixture, n_players: int, inference_server_type: str
    ):
        if not ray.is_initialized():
            ray.init()

        agents = [f"player_{i}" for i in range(n_players)]
        manager = create_manager(
            mocker,
            stopping_conditions={"rollout": {"max_iteration": 2}},
            rollout_config={
                "fragment_length": 100,
                "max_step": 10,
                "num_eval_episodes": 2,
                "num_threads": 1,
                "num_env_per_thread": 1,
                "num_eval_threads": 1,
                "use_subproc_env": False,
                "batch_mode": "timestep",
                "postprocessor_types": None,
                "eval_interval": 2,
                "inference_server": inference_server_type,
            },
            env_desc={
                "possible_agents": agents,
                "observation_spaces": {
                    agent: spaces.Box(-1, 1.0, shape=(2,)) for agent in agents
                },
                "action_spaces": {
                    agent: spaces.Box(-1, 1, shape=(2,)) for agent in agents
                },
            },
        )

        strategy_specs = {
            agent: StrategySpec(
                identifier=agent,
                policy_ids=["policy_0"],
                meta_data={
                    "prob_list": [1.0],
                    "policy_cls": None,
                    "kwargs": None,
                    "experiment_tag": "test_rollout_manager",
                },
            )
            for agent in agents
        }
        task_list = [
            {
                "trainable_agents": agents,
                "data_entrypoints": None,
                "strategy_specs": strategy_specs,
            }
            for _ in range(2)
        ]
        manager.rollout(task_list)

        for result in manager.retrive_results():
            print(result)

        ray.shutdown()

    def test_simulation_task_send(
        self, mocker: MockerFixture, n_players: int, inference_server_type: str
    ):
        if not ray.is_initialized():
            ray.init()

        agents = [f"player_{i}" for i in range(n_players)]
        manager = create_manager(
            mocker,
            stopping_conditions={"rollout": {"max_iteration": 2}},
            rollout_config={
                "fragment_length": 100,
                "max_step": 10,
                "num_eval_episodes": 2,
                "num_threads": 1,
                "num_env_per_thread": 1,
                "num_eval_threads": 1,
                "use_subproc_env": False,
                "batch_mode": "timestep",
                "postprocessor_types": None,
                "eval_interval": 2,
                "inference_server": inference_server_type,
            },
            env_desc={
                "possible_agents": agents,
                "observation_spaces": {
                    agent: spaces.Box(-1, 1.0, shape=(2,)) for agent in agents
                },
                "action_spaces": {
                    agent: spaces.Box(-1, 1, shape=(2,)) for agent in agents
                },
            },
        )

        manager.simulate([None] * 2)
        for result in manager.retrive_results():
            print(result)
        ray.shutdown()
