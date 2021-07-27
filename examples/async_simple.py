"""
Async optimizer for single-agent RL algorithms running simple scenario from MPE enviroments. In this case, there will be more than one agent training interfaces
used to do policy learning in async mode. Users can specify the number with `--num_learner`.
"""

import argparse

from malib.envs import MPE
from malib.runner import run


parser = argparse.ArgumentParser("Async training on mpe environments.")

parser.add_argument(
    "--num_learner",
    type=int,
    default=3,
    help="The number of agent training interfaces. Default by 3.",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Trianing batch size. Default by 64."
)
parser.add_argument(
    "--num_epoch", type=int, default=100, help="Training epoch. Default by 100."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="DQN",
    help="The single-agent RL algortihm registered in MALib. Default by DQN",
)


if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {
        "scenario_configs": {"max_cycles": 25},
        "env_id": "simple_v2",
    }
    env = MPE(**env_config)
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    run(
        group="MPE/simple",
        name="async_dqn",
        env_description={
            "creator": MPE,
            "config": env_config,
            "possible_agents": possible_agents,
        },
        agent_mapping_func=lambda agent: [
            f"{agent}_async_{i}" for i in range(args.num_learner)
        ],
        training={
            "interface": {
                "type": "async",
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "population_size": -1,
            },
            "config": {
                "update_interval": 1,
                "saving_interval": 10,
                "batch_size": args.batch_size,
                "num_epoch": 100,
                "return_gradients": True,
            },
        },
        algorithms={
            "Async": {"name": args.algorithm},
        },
        rollout={
            "type": "async",
            "stopper": "simple_rollout",
            "stopper_config" "metric_type": "simple",
            "fragment_length": env_config["scenario_configs"]["max_cycles"],
            "num_episodes": 100,  # episode for each evaluation/training epoch
            "terminate": "any",
        },
        global_evaluator={
            "name": "generic",
            "config": {"stop_metrics": {}},
        },
        dataset_config={"episode_capacity": 30000},
    )
