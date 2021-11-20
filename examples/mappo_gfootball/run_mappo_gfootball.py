# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch

import yaml
import os

from rollout_function import grf_simultaneous
from malib.runner import run
import copy
from malib.utils.logger import Logger
import pathlib
from malib.envs.gr_football import build_sampler_config, env_desc_gen

BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("General training on Google Research Football.")
    parser.add_argument(
        "--config", type=str, help="YAML configuration path.", required=True
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no_popart", action="store_true", default=False)
    parser.add_argument("--no_feature_norm", action="store_true", default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    Logger.info(f"the seed is set to be {args.seed}.")

    with open(args.config, "r") as f:
        config = yaml.load(f)


    training_config = config["training"]
    rollout_config = config["rollout"]

    assert (
        rollout_config["num_episodes"] % rollout_config["episode_seg"] == 0
    ), "in rollout config (num_episodes mod epsidoe_seg) must be 0"
    training_config["config"]["total_epoch"] = rollout_config["stopper_config"][
        "max_step"
    ]

    rollout_config["callback"] = grf_simultaneous
    evaluation_config = config["evaluation"]
    evaluation_config["callback"] = grf_simultaneous
    env_desc = env_desc_gen(config=config["env_description"]["config"])

    training_config["interface"]["observation_spaces"] = env_desc["observation_spaces"]
    training_config["interface"]["action_spaces"] = env_desc["action_spaces"]

    custom_config = config["algorithms"]["MAPPO"]["custom_config"]
    custom_config["use_popart"] = not args.no_popart
    custom_config["use_feature_normalization"] = not args.no_feature_norm
    # FOR MAPPO
    custom_config.update({"global_state_space": env_desc["state_spaces"]})


    model_config = config["algorithms"]["MAPPO"]["model_config"]
    build_sampler_config(
        env_desc,
        model_config["actor"]["layers"][-1]["units"],
        model_config["critic"]["layers"][-1]["units"],
    )


    run(
        group=config["group"],
        name=config["name"],
        env_description=env_desc,
        agent_mapping_func=lambda agent: agent[
            :6
        ],  # e.g. "team_0_player_0" -> "team_0"
        training=training_config,
        algorithms=config["algorithms"],
        rollout=rollout_config,
        evaluation=config.get("evaluation", {}),
        global_evaluator=config["global_evaluator"],
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
        # "nash", "fictitious_self_play"
        # solver="nash",
        worker_config=config["worker_config"],
        use_init_policy_pool=False
    )
