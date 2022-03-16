import os
import time
import psutil
import tabulate
import pytest
import threading
import ray

from itertools import product

from malib import settings
from malib.utils.typing import List

from malib.envs import gr_football
from malib.envs.vector_env import VectorEnv, SubprocVecEnv
from malib.envs.agent_interface import AgentInterface

from malib.algorithm.random import RandomPolicy
from malib.algorithm.mappo import MAPPO
from malib.backend.datapool import parameter_server

from tests.football.rollout_case import (
    run_simple_loop,
    run_simple_ray_pool,
    run_vec_env,
)
from tests.football.training_case import gen_policy_description

# from tests.parameter_server import FakeParameterServer
from tests.dataset import FakeDataServer


@pytest.fixture(scope="session")
def env_desc():

    return gr_football.env_desc_gen(
        {
            "env_id": "PSGFootball",
            "use_built_in_GK": True,
            "scenario_configs": {
                "env_name": "5_vs_5",
                "number_of_left_players_agent_controls": 4,
                "number_of_right_players_agent_controls": 0,
                "representation": "raw",
                "stacked": False,
                "logdir": "/tmp/football/malib_psro",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
            },
        }
    )


@pytest.fixture(scope="session")
def servers():
    if not ray.is_initialized():
        ray.init()

    ParameterServer = parameter_server.ParameterServer.as_remote()
    pserver = ParameterServer.options(name=settings.PARAMETER_SERVER_ACTOR).remote()
    dataset_server = FakeDataServer.options(
        name=settings.OFFLINE_DATASET_ACTOR
    ).remote()

    return (pserver, dataset_server)


@pytest.mark.parametrize(
    "vec_modes, num_envs_list",
    [
        (["ray_vec"], [1, 2, 4, 8, 16, 32, 64, 128]),
    ],
)
def test_vec_env_performance(
    env_desc, servers, vec_modes: List[str], num_envs_list: List[int]
):
    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]

    table = [None] * (len(vec_modes) * len(num_envs_list))
    cur_pid = os.getpid()
    results = [None] * (len(vec_modes) * len(num_envs_list))

    # create a thread for monitor
    done = False

    print("start")

    def monitor(idx):
        interval = 2
        ave_cpu_percent = 0.0
        ave_mem_percent = 0.0
        count = 0

        while not done:
            ave_cpu_percent += psutil.cpu_percent()
            ave_mem_percent += psutil.virtual_memory().percent
            count += 1
            time.sleep(interval)

        count = max(1, count)

        results[idx] = [ave_cpu_percent / count, ave_mem_percent / count]

    for i, (vec_mode, num_envs) in enumerate(product(vec_modes, num_envs_list)):
        done = False
        if vec_mode == "ray_vec":
            # start training process here
            policy_template = RandomPolicy(
                "random",
                observation_space=obs_spaces["team_0"],
                action_space=act_spaces["team_0"],
                model_config=None,
                custom_config=None,
            )
            policy_description = gen_policy_description(policy_template, env_desc)
            # push to parameter server
            for (_, _, param_desc) in policy_description.values():
                ray.get(servers[0].push.remote(parameter_desc=param_desc))
            N_Frames, FPS = run_vec_env(
                env_desc,
                num_envs,
                runtime_configs={
                    "num_episodes": num_envs,
                    "num_env_per_worker": 1,
                    "use_subproc_env": False,
                    "batch_mode": "time_step",
                    "fragment_length": num_envs * 3000,
                    "flags": "evaluation",
                    "policy_description": policy_description,
                    "trainable_pairs": {"team_0": ("policy_0", _)},
                    "policy_combinations": [{"team_0": "policy_0"}],
                },
            )
        elif vec_mode == "simple_loop":
            N_Frames, FPS = run_simple_loop(env_desc, num_envs)
        elif vec_mode == "simple_ray_pool":
            N_Frames, FPS = run_simple_ray_pool(env_desc, num_envs)
        elif vec_mode == "subproc":
            vec_env = SubprocVecEnv(
                observation_spaces=obs_spaces,
                action_spaces=act_spaces,
                creator=env_desc["creator"],
                configs=env_desc["configs"],
                max_num_envs=num_envs,
            )

        # run and profilling
        thread = threading.Thread(target=monitor, args=(i,))
        thread.start()

        done = True

        # vec_type, num_envs, FPS, CPU ratio, MEM ratio
        thread.join()
        table[i] = [vec_mode, num_envs, N_Frames, FPS] + results[i]
        print(table[i])

    print("")

    print(
        tabulate.tabulate(
            table,
            headers=[
                "Vec Type",
                "Num of Envs",
                "Frames",
                "FPS",
                "CPU Ratio",
                "MEM Ratio",
            ],
        )
    )
