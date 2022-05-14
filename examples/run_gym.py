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
        config = argparse.Namespace(**yaml.safe_load(f))

    run(
        task_mode="marl",
        group=config.group,
        name=config.name,
        env_description=config.env_description,
        agent_mapping_func=lambda agent: "share",
        training=config.training,
        algorithms=config.algorithms,
        rollout_worker=config.rollout_worker,
        evaluation=config.evaluation,
        global_evaluator=config.global_evaluator,
        dataset_config=config.dataset_config,
    )
