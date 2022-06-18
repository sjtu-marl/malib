from argparse import ArgumentParser

import os
import time
import shutup

shutup.please()

from malib.runner import run
from malib.agent import IndependentAgent
from malib.scenarios.marl_scenario import MARLScenario
from malib.algorithm.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG
from malib.rollout.envs.gr_football import env_desc_gen


if __name__ == "__main__":
    parser = ArgumentParser(
        "Multi-agent reinforcement learning for Google Research Football"
    )
    parser.add_argument(
        "--use_group", action="store_true", help="enable agent grouping or not."
    )
    parser.add_argument("--log_dir", default="./logs/", help="Log directory.")

    args = parser.parse_args()

    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    trainer_config["total_timesteps"] = int(1e6)

    training_config = {
        "type": IndependentAgent,
        "trainer_config": trainer_config,
        "custom_config": {},
    }
    max_step = 300
    num_env_per = 10
    rollout_config = {
        "fragment_length": max_step * num_env_per,  # every thread
        "max_step": max_step,
        "num_eval_episodes": num_env_per,
        "num_threads": 1,
        "num_env_per_thread": num_env_per,
        "num_eval_threads": 1,
        "use_subproc_env": False,
        "batch_mode": "time_step",
        "postprocessor_types": ["defaults"],
        # every # rollout epoch run evaluation.
        "eval_interval": 1,
        "inference_server": "ray",  # three kinds of inference server: `local`, `pipe` and `ray`
    }
    agent_mapping_func = lambda agent: agent

    algorithms = {
        "default": (
            DQNPolicy,
            DQNTrainer,
            # model configuration, None for default
            {},
            {},
        )
    }

    env_description = env_desc_gen(None, group=args.use_group)
    runtime_logdir = os.path.join(args.log_dir, f"grf_dqn/{time.time()}")

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
            "training": {"max_iteration": int(1e10)},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )

    run(scenario)
