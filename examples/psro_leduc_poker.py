# -*- coding: utf-8 -*-
import argparse
import datetime
import os

# from pettingzoo.classic import leduc_holdem_v2 as leduc_holdem
# https://www.pettingzoo.ml/classic/leduc_holdem
from malib import settings
from malib.envs import PokerEnv
from malib.runner import run

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser("PSRO training on mpe environments.")

parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--fragment_length", type=int, default=10240)
parser.add_argument("--worker_num", type=int, default=3)
parser.add_argument("--log_dir", type=str, default=settings.LOG_DIR)
parser.add_argument("--algorithm", type=str, default="PPO", choices={"PPO", "DQN"})

args = parser.parse_args()


num_total_training_episode = 55

# args.algorithm = "SAC"
args.batch_size = 128
args.num_epoch = 8
args.fragment_length = 10
num_episode = 1000

# TODO(ming): add to offline dataset
off_policy_start_sample_timestamp = 500
buffer_size = 200000
num_simulation = 100

now_time = now = datetime.datetime.now(
    tz=datetime.timezone(datetime.timedelta(hours=8))
)
args.log_dir = os.path.join(
    # settings.LOG_DIR, f"DQN/{now_time.strftime('%Y%m%d_%H_%M_%S')}"
    settings.LOG_DIR,
    f"SAC/{now_time.strftime('%Y%m%d_%H_%M_%S')}",
)
args.work_num = 3


if __name__ == "__main__":
    env_description = {
        "creator": PokerEnv,
        "config": {"scenario_configs": {"fixed_player": True}, "env_id": "leduc"},
    }

    env = PokerEnv(**env_description["config"])
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    env_description["possible_agents"] = possible_agents

    run(
        group="psro",
        name="leduc_poker",
        env_description=env_description,
        training={
            "interface": {
                "type": "independent",
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
            },
            "config": {
                "batch_size": args.batch_size,
                "update_interval": args.num_epoch,
            },
        },
        algorithms={
            args.algorithm: {
                "name": args.algorithm,
                "custom_config": {
                    "gamma": 1.0,
                    "eps_min": 0,
                    "eps_max": 1.0,
                    "eps_decay": 100,
                    "lr": 1e-2,
                },
            }
        },
        rollout={
            "type": "async",
            "stopper": "simple_rollout",
            "stopper_config": {"max_step": num_total_training_episode},
            "metric_type": "simple",
            "fragment_length": args.fragment_length,
            "num_episodes": num_episode,
            "episode_seg": 100,
        },
        evaluation={
            "max_episode_length": 5,
            "num_episode": num_simulation,
        },  # dict(num_simulation=num_simulation, sim_max_episode_length=5),
        global_evaluator={
            "name": "psro",
            "config": {
                "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
            },
        },
        dataset_config={"episode_capacity": buffer_size},
    )
