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

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.safe_load(f)

    run(use_init_policy_pool=False, **config)
