"""
Usage:

```bash
cd malib
export PYTHONPATH=./malib/envs/smarts/_env:$PYTHONPATH
python examples/run_smarts.py --config examples/configs/smarts/ddpg.yaml
```
"""

import argparse
from malib.backend.datapool.offline_dataset_server import Episode
import os
import yaml

from malib.runner import run
from malib.envs import SMARTS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on SMARTS marl benchmarking.")
    parser.add_argument(
        "--config", type=str, default="examples/configs/smarts/ddpg.yaml"
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.safe_load(f)

    env_desc = config["env_description"]
    env_desc["cretor"] = SMARTS

    env = SMARTS(**env_desc["config"])

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    scenarios = env.scenarios

    print(
        "observation_spaces: {}\naction spaces: {}\nscenario: {}\nagents: {}".format(
            observation_spaces, action_spaces, scenarios, possible_agents
        )
    )

    done = False
    cnt = 0
    obs = env.reset()
    while not done:
        actions = {agent: space.sample() for agent, space in action_spaces.items()}
        rets = env.step(actions)
        rewards = rets[Episode.REWARD]
        done = any(rets[Episode.DONE].values())
        print("step: {}, actions: {}, rewards: {}".format(cnt, actions, rewards))
        cnt += 1
    env.close()
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
