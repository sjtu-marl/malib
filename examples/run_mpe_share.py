"""
agent mapping func = lambda agent: share
"""

import argparse
import yaml
import os

from malib.envs import MPE
from malib.runner import run


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on Atari games.")
    parser.add_argument(
        "--config", type=str, help="YAML configuration path.", required=True
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable parameter sharing or not."
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = argparse.Namespace(**yaml.safe_load(f))

    agent_mapping_func = (
        (lambda agent: agent) if not args.share else (lambda agent: "agent")
    )
    algo = list(config.algorithms.keys())[0]

    run(
        group="MPE",
        name=f"share={args.share}_eid={config.env_description['config']['env_id']}_algo={algo}",
        # parameter sharing, will derive only one agent
        agent_mapping_func=agent_mapping_func,
        task_mode="marl",
        env_description=config.env_description,
        training=config.training,
        algorithms=config.algorithms,
        rollout_worker=config.rollout_worker,
        evaluation=config.evaluation,
        global_evaluator=config.global_evaluator,
        dataset_config=config.dataset_config,
    )
