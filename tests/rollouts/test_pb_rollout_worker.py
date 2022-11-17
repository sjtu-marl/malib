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
from malib.remote.interface import RemoteInterface


class Fakeclient(RemoteInterface):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server: ray.ObjectRef,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: Any = None,
        custom_config: Dict[str, Any] = {},
    ):
        self.max_env_num = max_env_num
        self.agents = env_desc["possible_agents"]

    def add_envs(self, maxinum: int) -> int:
        return self.max_env_num

    def close(self):
        time.sleep(0.5)
        return

    def run(
        self,
        agent_interfaces,
        rollout_config,
        dataset_writer_info_dict=None,
    ) -> Dict[str, Any]:
        time.sleep(0.5)
        return {
            "evaluation": {f"agent_reward/{agent}_mean": 1.0 for agent in self.agents},
            "total_timesteps": 1000,
            "FPS": 100000,
        }


class FakeServer(RemoteInterface):
    def __init__(
        self,
        agent_id,
        observation_space,
        action_space,
        parameter_server,
        governed_agents,
    ) -> None:
        pass

    def shutdown(self):
        time.sleep(0.5)
        return

    def save(self, model_dir: str):
        print("called save method")
        return

    def compute_action(self, dataframes, runtime_config):
        print("called computation action")
        return None


def create_rollout_worker(
    mocker: MockerFixture, env_desc: Dict[str, Any], rollout_config: Dict[str, Any]
):
    mocker.patch("malib.rollout.rolloutworker.RayInferenceClient", new=Fakeclient)
    mocker.patch("malib.rollout.rolloutworker.RayInferenceServer", new=FakeServer)
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
        )

        ray.kill(parameter_server)
        ray.kill(dataset_server)

        ray.shutdown()

    #     rollout_thread = threading.Thread(
    #         target=worker.rollout,
    #         args=(
    #             runtime_strategy_specs,
    #             stopping_conditions,
    #             data_entrypoints,
    #             trainable_agents,
    #         ),
    #     )
    #     rollout_thread.start()
    #     # wait for 5 seconds
    #     time.sleep(5)
    #     worker.stop_pending_tasks()
    #     rollout_thread.join()
    #     ray.shutdown()

    # def test_simulation(self):
    #     if not ray.is_initialized():
    #         ray.init()
    #     worker = create_rollout_worker(env_desc, rollout_config)
    #     worker.simulate(runtime_strategy_specs_list)
    #     ray.shutdown()
