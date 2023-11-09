import os
import time

from argparse import ArgumentParser

from malib.learner import IndependentAgent
from malib.scenarios.marl_scenario import MARLScenario

from malib.runner import run
from malib.rl.ppo import PPOPolicy, PPOTrainer, DEFAULT_CONFIG
from malib.rollout.envs.gym import env_desc_gen


if __name__ == "__main__":
    parser = ArgumentParser("Use PPO solve Gym tasks.")
    parser.add_argument("--log-dir", default="./logs/", help="Log directory.")
    parser.add_argument("--env-id", default="CartPole-v1", help="gym environment id.")
    parser.add_argument("--use-cuda", action="store_true")

    args = parser.parse_args()

    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    trainer_config["total_timesteps"] = int(1e6)
    trainer_config["use_cuda"] = args.use_cuda

    training_config = {
        "learner_type": IndependentAgent,
        "trainer_config": trainer_config,
        "custom_config": {},
    }

    rollout_config = {
        "fragment_length": 2000,  # determine the size of sended data block
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

    # one to one, no sharing, if sharing, implemented as:
    #   agent_mapping_func = lambda agent: "default"
    agent_mapping_func = lambda agent: agent

    algorithms = {
        "default": (
            PPOPolicy,
            PPOTrainer,
            # model configuration, None as default
            {},
            {"use_cuda": args.use_cuda},
        )
    }

    env_description = env_desc_gen(env_id=args.env_id, scenario_configs={})
    runtime_logdir = os.path.join(args.log_dir, f"sa_ppo_gym/{time.time()}")

    if not os.path.exists(runtime_logdir):
        os.makedirs(runtime_logdir)

    scenario = MARLScenario(
        name=f"ppo-gym-{args.env_id}",
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
