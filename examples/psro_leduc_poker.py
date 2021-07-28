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

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epoch", type=int, default=8)
parser.add_argument("--fragment_length", type=int, default=100)
parser.add_argument("--worker_num", type=int, default=3)
parser.add_argument("--algorithm", type=str, default="DQN")
parser.add_argument("--num_total_training_episode", type=int, default=55)
parser.add_argument("--num_episode", type=int, default=1000)
parser.add_argument("--buffer_size", type=int, default=200000)
parser.add_argument("--num_simulation", type=int, default=100)
parser.add_argument("--episode_seg", type=int, default=100)

args = parser.parse_args()

if __name__ == "__main__":
    env_description = {
        "creator": PokerEnv,
        "config": {
            "scenario_configs": {"fixed_player": True},
            "env_id": "leduc_poker",
        },
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
            "stopper_config": {"max_step": args.num_total_training_episode},
            "metric_type": "simple",
            "fragment_length": args.fragment_length,
            "num_episodes": args.num_episode,
            "episode_seg": args.episode_seg,
        },
        evaluation={
            "max_episode_length": 100,
            "num_episode": args.num_simulation,
        },  # dict(num_simulation=num_simulation, sim_max_episode_length=5),
        global_evaluator={
            "name": "psro",
            "config": {
                "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
            },
        },
        dataset_config={"episode_capacity": args.buffer_size},
    )
