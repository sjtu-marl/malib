"""
agent mapping func = lambda agent: share
"""

import argparse

import yaml
import os

from malib.envs import gym as custom_gym
from malib.runner import run
from malib.utils.preprocessor import get_preprocessor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "General training on single-agent Gym environments."
    )
    parser.add_argument(
        "--config", type=str, help="YAML configuration path.", required=True
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    # read environment description
    env_desc = custom_gym.env_desc_gen(config["env_description"]["config"]["env_id"])
    obs_space_template = list(env_desc["observation_spaces"].values())[0]
    preprocessor = get_preprocessor(space=obs_space_template)(obs_space_template)
    sampler_config = custom_gym.basic_sampler_config(
        env_desc["observation_spaces"], env_desc["action_spaces"], preprocessor
    )

    agent_wise_shapes = sampler_config["data_shapes"]
    env_desc["config"]["data_shapes"] = dict.fromkeys(
        env_desc["possible_agents"], agent_wise_shapes
    )

    training_config = config["training"]
    rollout_config = config["rollout"]

    training_config["interface"]["observation_spaces"] = env_desc["observation_spaces"]
    training_config["interface"]["action_spaces"] = env_desc["action_spaces"]

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
        global_evaluator=config["global_evaluator"],
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
    )
