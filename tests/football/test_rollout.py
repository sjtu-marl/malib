import os
import time
import psutil
import tabulate
import pytest
import threading
import ray
import numpy as np

from itertools import product

from malib.utils.typing import List
from malib.utils.episode import EpisodeKey
from malib.envs import gr_football
from malib.envs.vector_env import VectorEnv, SubprocVecEnv


@ray.remote(num_cpus=0)
class _RemoteEnv:
    def __init__(self, creater, env_config) -> None:
        self.env = creater(**env_config)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        return ret

    def step(self, action):
        ret = self.env.step(action)
        return ret


def act_func(obs, act_space):
    return np.asarray([act_space.sample() for _ in obs])


def run_simple_loop(env_desc, num):
    envs = [env_desc["creator"](**env_desc["config"]) for _ in range(num)]
    action_spaces = env_desc["action_spaces"]

    rets = [env.reset() for env in envs]
    for i, ret in enumerate(rets):
        rets[i][EpisodeKey.NEXT_OBS] = ret[EpisodeKey.CUR_OBS]

    done = False
    step_limit = 3000

    cnt = 0
    n_frames = 0

    start_time = time.time()
    while len(envs) > 0 and cnt < step_limit:
        new_envs = []
        new_rets = []
        for i, env in enumerate(envs):
            actions = {
                aid: act_func(obs, action_spaces[aid])
                for aid, obs in rets[i][EpisodeKey.NEXT_OBS].items()
            }
            ret = env.step(actions)
            done = ret[EpisodeKey.DONE]["__all__"]
            if not done:
                new_envs.append(env)
                new_rets.append(ret)
            else:
                n_frames += cnt
        envs = new_envs
        rets = new_rets
        cnt += 1
    end_time = time.time()

    if len(envs) > 0:
        n_frames += cnt * len(envs)

    return n_frames, n_frames / (end_time - start_time)


def run_simple_ray_pool(env_desc, num):
    creator = env_desc["creator"]
    config = env_desc["config"]
    action_spaces = env_desc["action_spaces"]
    # create remote env

    envs = [_RemoteEnv.remote(creater=creator, env_config=config) for _ in range(num)]
    rets = ray.get([env.reset.remote() for env in envs])
    for i, ret in enumerate(rets):
        rets[i][EpisodeKey.NEXT_OBS] = ret[EpisodeKey.CUR_OBS]

    done = False
    step_limit = 3000

    cnt = 0
    n_frames = 0

    start_time = time.time()
    while len(envs) > 0 and cnt < step_limit:
        new_envs = []
        new_rets = []
        actions = {
            aid: act_func(obs, action_spaces[aid])
            for aid, obs in rets[i][EpisodeKey.NEXT_OBS].items()
        }
        rets_vec = ray.get([env.step.remote(actions) for env in envs])
        for i, ret in enumerate(rets_vec):
            done = ret[EpisodeKey.DONE]["__all__"]
            if not done:
                new_envs.append(envs[i])
                new_rets.append(ret)
            else:
                n_frames += cnt
        envs = new_envs
        rets = new_rets
        cnt += 1
    end_time = time.time()

    if len(envs) > 0:
        n_frames += cnt * len(envs)

    return n_frames, n_frames / (end_time - start_time)


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


@pytest.mark.parametrize(
    "vec_modes, num_envs_list",
    [
        (["simple_loop", "simple_ray_pool"], [1, 2, 4, 8, 16, 32, 64, 128]),
    ],
)
def test_vec_env_performance(env_desc, vec_modes: List[str], num_envs_list: List[int]):
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
            vec_env = VectorEnv(
                observation_spaces=obs_spaces,
                action_spaces=act_spaces,
                creator=env_desc["creator"],
                configs=env_desc["configs"],
                preset_num_envs=num_envs,
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
