from argparse import ArgumentParser

import os
import time
import shutup

shutup.please()

from malib.runner import run
from malib.agent import IndependentAgent
from malib.scenarios.marl_scenario import MARLScenario
from malib.algorithm.pg import PGPolicy, PGTrainer
from malib.rollout.envs.gym import env_desc_gen


if __name__ == "__main__":
    parser = ArgumentParser("Multi-agent reinforcement learning.")
    parser.add_argument("--log_dir", default="./logs/", help="Log directory.")
    parser.add_argument("--env_id", default="CartPole-v1", help="gym environment id.")

    args = parser.parse_args()

    training_config = {
        "type": IndependentAgent,
        "trainer_config": {
            "use_cuda": False,
            "batch_size": 32,
            "optimizer": "Adam",
            "lr": 1e-4,
            "reward_norm": None,
            "gamma": 0.99,
        },
        "custom_config": {},
    }
    rollout_config = {
        "fragment_length": 2000,  # every thread
        "max_step": 200,
        "num_eval_episodes": 10,
        "num_threads": 2,
        "num_env_per_thread": 10,
        "num_eval_threads": 1,
        "use_subproc_env": False,
        "batch_mode": "time_step",
        "postprocessor_types": ["defaults"],
        # every # rollout epoch run evaluation.
        "eval_interval": 10,
        "inference_server": "local",  # three kinds of inference server: `local`, `pipe` and `ray`
    }
    agent_mapping_func = lambda agent: agent

    algorithms = {
        "default": (
            PGPolicy,
            PGTrainer,
            # model configuration, None for default
            {},
            {},
        )
    }

    env_description = env_desc_gen(env_id=args.env_id, scenario_configs={})
    runtime_logdir = os.path.join(args.log_dir, f"gym/{time.time()}")

    if not os.path.exists(runtime_logdir):
        os.makedirs(runtime_logdir)

    scenario = MARLScenario(
        name="gym",
        log_dir=runtime_logdir,
        algorithms=algorithms,
        env_description=env_description,
        training_config=training_config,
        rollout_config=rollout_config,
        agent_mapping_func=agent_mapping_func,
        stopping_conditions={
            "training": {"max_iteration": 1000},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )

    run(scenario)
