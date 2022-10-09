from argparse import ArgumentParser

import os
import time
import shutup

shutup.please()

from malib.runner import run
from malib.agent import IndependentAgent
from malib.scenarios.psro_scenario import PSROScenario
from malib.rl.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG
from malib.rollout.envs.dummy_env import env_desc_gen


if __name__ == "__main__":
    parser = ArgumentParser("Naive Policy Space Response Oracles.")
    parser.add_argument("--log_dir", default="./logs/", help="Log directory.")

    args = parser.parse_args()

    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    trainer_config["total_timesteps"] = int(1e6)

    training_config = {
        "type": IndependentAgent,
        "trainer_config": trainer_config,
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

    env_description = env_desc_gen()
    runtime_logdir = os.path.join(args.log_dir, f"psro_dummy/{time.time()}")

    if not os.path.exists(runtime_logdir):
        os.makedirs(runtime_logdir)

    scenario = PSROScenario(
        name="psro_dummpy",
        log_dir=runtime_logdir,
        algorithms=algorithms,
        env_description=env_description,
        training_config=training_config,
        rollout_config=rollout_config,
        # control the outer loop.
        global_stopping_conditions={"max_iteration": 2},
        agent_mapping_func=agent_mapping_func,
        # for the training of best response.
        stopping_conditions={
            "training": {"max_iteration": int(1e2)},
            "rollout": {"max_iteration": 2},
        },
    )

    run(scenario)
