"""
agent mapping func = lambda agent: share
"""

import argparse

import yaml
import os
import numpy as np

from malib.runner import run
from malib.utils.metrics import get_metric
from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs import SC2Env


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on SC2 games.")
    parser.add_argument(
        "--config",
        type=str,
        help="YAML configuration path.",
        default="examples/configs/sc2/qmix.yaml",
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    env_desc = config["env_description"]
    env_desc["creator"] = SC2Env
    env = SC2Env(**env_desc["config"])
    config["rollout"]["fragment_length"] = env.env_info["episode_limit"]

    teams = {}
    for aid in env.possible_agents:
        tid = "SC2"  # for star craft
        if tid not in teams.keys():
            teams[tid] = []
        teams[tid].append(aid)

    env_desc["teams"] = teams

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    print(f"observation spaces: {observation_spaces}")
    print(f"action spaces: {action_spaces}")
    print(f"state spaces: {env.global_state_space}")

    env_desc["possible_agents"] = env.possible_agents
    env.close()

    training_config = config["training"]
    rollout_config = config["rollout"]

    training_config["interface"]["observation_spaces"] = observation_spaces
    training_config["interface"]["action_spaces"] = action_spaces
    for algo in config["algorithms"].values():
        algo["custom_config"]["global_state_space"] = env.global_state_space

    run(
        group=config["group"],
        name=config["name"],
        env_description=env_desc,
        agent_mapping_func=lambda agent: "share",
        training=training_config,
        algorithms=config["algorithms"],
        # rollout configuration for each learned policy model
        rollout=rollout_config,
        evaluation=config.get("evaluation", {}),
        global_evaluator=config.get("global_evaluator", {}),
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
    )
