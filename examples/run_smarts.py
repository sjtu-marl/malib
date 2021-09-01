import argparse
from re import M
from malib.envs.smarts.common import ObservationAdapter
import os
import yaml

from examples.run_sc2_share import BASE_DIR

from malib.runner import run
from malib.envs import SMARTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on SMARTS marl benchmarking.")
    parser.add_argument(
        "--config", type=str, default="examples/configs/smarts/ppo.yaml"
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    env_desc = config["env_description"]
    env_desc["cretor"] = SMARTS

    env = SMARTS(**env_desc["config"])

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    scenarios = env.scenarios

    print(
        "observation_spaces: {}\naction spaces: {}\nscenario: {}".format(
            observation_spaces, action_spaces, scenarios
        )
    )

    done = False
    cnt = 0
    # while not done:

    # training_config = config["training"]
    # rollout_config = config["rollout"]

    # run(
    #     group=config["group"],
    #     name=config["name"],
    #     env_description=env_desc,
    #     agent_mapping_func=lambda agent: "share",
    #     training=training_config,
    #     rollout=rollout_config,
    # )
