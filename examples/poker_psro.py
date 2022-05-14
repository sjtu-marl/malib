# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

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

import argparse
import os

from malib.runner import run
from malib.envs.poker import env_desc_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solve poker games with psro.")
    parser.add_argument(
        "--env_id", type=str, required=True, choices={"leduc_poker", "kuhn_poker"}
    )
    parser.add_argument(
        "--br",
        type=str,
        default="DQN",
        help="Algorithm for training best response, default is DQN.",
    )
    args = parser.parse_args()

    env_desc = env_desc_gen(env_id=args.env_id, scenario_configs={"fixed_player": True})
    tmp_env = env_desc["creator"](**env_desc["config"])

    run(
        group="poker",
        name=f"{args.env_id}_psro",
        env_description=env_desc,
        training={
            "interface": {
                "type": "independent",
                "observation_spaces": tmp_env.observation_spaces,
                "action_spaces": tmp_env.action_spaces,
                "use_init_policy_pool": True,
            },
            "config": {
                "use_cuda": True,
                "update_interval": 1,
                "saving_interval": 10,
                "batch_size": 64,
                "optimizer": "Adam",
                "lr": 0.01,
                "tau": 0.01,
            },
        },
        algorithms={
            "best_response": {
                "name": args.br,
                "custom_config": {
                    "gamma": 1.0,
                    "eps_min": 0.1,
                    "eps_max": 1.0,
                    "eps_anneal_time": 100,
                    "lr": 1e-2,
                },
            }
        },
        rollout_worker={
            "callback": "sequential",
            "stopper": {"name": "simple_rollout", "config": {"max_step": 1250}},
            "num_threads": 4,
            "num_env_per_thred": 2,
            "num_eval_threads": 5,
            "batch_mode": "time_step",
            "post_processor_types": ["default"],
            "use_subproc_env": False,
            "task_config": {"max_step": 20, "fragment_length": 200},
        },
        evaluation={"fragment_length": 200, "num_episodes": 1},
        global_evaluator={
            "name": "psro",
            "config": {
                "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
            },
        },
        dataset_config={"episode_capacity": int(1e4)},
        task_mode="gt",
    )
