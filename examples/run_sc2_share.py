"""
agent mapping func = lambda agent: share
"""

import argparse
import traceback

import yaml
import os
import time

from malib.runner import run
from malib.utils.logger import Logger
from malib.envs import sc_desc_gen


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

    env_desc = sc_desc_gen(**config["env_description"]["config"])
    config["rollout"]["fragment_length"] = 100

    training_config = config["training"]
    rollout_config = config["rollout"]

    training_config["interface"]["observation_spaces"] = env_desc["observation_spaces"]
    training_config["interface"]["action_spaces"] = env_desc["action_spaces"]
    for algo in config["algorithms"].values():
        algo["custom_config"]["global_state_space"] = list(
            env_desc["global_state_spaces"].values()
        )[0]

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

    code = 0
    while code == 0:
        code = os.system("kill -9 $(ps -ef | grep SC2 | grep -v grep|awk '{print $2}')")
        time.sleep(1)
    Logger.info("All SC2 processes have been terminated with code={}".format(code))
