# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pragma: no cover
from argparse import ArgumentParser

import os
import time

from malib.runner import run
from malib.agent import IndependentAgent
from malib.scenarios.marl_scenario import MARLScenario
from malib.rl.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG
from malib.rollout.envs.gym import env_desc_gen


if __name__ == "__main__":
    parser = ArgumentParser("Multi-agent reinforcement learning for gym cases.")
    parser.add_argument("--log-dir", default="./logs/", help="Log directory.")
    parser.add_argument("--env-id", default="CartPole-v1", help="gym environment id.")
    parser.add_argument("--use-cuda", action="store_true")

    args = parser.parse_args()

    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    trainer_config["total_timesteps"] = int(1e6)
    trainer_config["use_cuda"] = args.use_cuda

    training_config = {
        "type": IndependentAgent,
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
    agent_mapping_func = lambda agent: agent

    algorithms = {
        "default": (
            DQNPolicy,
            DQNTrainer,
            # model configuration, None for default
            {},
            {"use_cuda": args.use_cuda},
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
            "training": {"max_iteration": int(1e10)},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )

    run(scenario)
