"""
Share an agent interface
"""
import argparse

import yaml
import os
import importlib
import supersuit

from typing import Dict, Sequence


from malib.envs.maatari.env import MAAtari
from malib.runner import run


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on Atari games.")
    parser.add_argument(
        "--config", type=str, help="YAML configuration path.", required=True
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    experiment_group = config["experiment_group"]
    env_desc = config["env_description"]
    env_desc["config"] = env_desc.get("config", {})
    # load creator
    env_desc["creator"] = MAAtari
    env = MAAtari(**env_desc["config"])

    traianable_agents = env.trainable_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    print(f"---------- observation spaces: {observation_spaces}")
    print(f"---------- action spaces: {action_spaces}")

    env_desc["possible_agents"] = env.trainable_agents
    env.close()

    training_config = config["training"]
    rollout_config = config["rollout"]

    training_config["interface"]["observation_spaces"] = observation_spaces
    training_config["interface"]["action_spaces"] = action_spaces

    run(
        experiment=experiment_group,
        env_description=env_desc,
        # generate multiple agent interfaces (one to many)
        agent_mapping_func=config.get(
            "agent_mapping_func", lambda agent: "share_agent"
        ),
        training=training_config,
        algorithms=config["algorithms"],
        # rollout configuration for each learned policy model
        rollout=rollout_config,
        evaluation=config.get("evaluation", {}),
        global_evaluator=config["global_evaluator"],
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
    )
