"""
Sample Expert Demonstrations in Gym Environments.
"""

import argparse
import os
import pprint
import numpy as np

import ray
import yaml

from malib import settings

from malib.agent import get_training_agent
from malib.rollout import rollout_func
from malib.utils import logger
from malib.utils.logger import Log
from malib.envs import GymEnv
from malib.rollout.rollout_worker import RolloutWorker
from malib.backend.datapool.offline_dataset_server import Episode, Table

# from malib.backend.datapool.offline_dataset_server import OfflineDataset, Table, Episode

import pickle as pkl


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(
    "Single instance of MARL training on gym environments."
)
parser.add_argument(
    "--config", type=str, help="YAML configuration path.", required=True
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the policy model used for sampling.",
    required=True,
)
parser.add_argument(
    "--sample_num",
    type=int,
    help="Num of samples in the demonstration data.",
    default=5000,
)
parser.add_argument(
    "--parallel_num",
    type=int,
    default=5,
)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    # ================== Clean configuration =================================================================
    env_desc = config["env_description"]
    env_desc["config"] = env_desc.get("config", {})
    env_desc["creator"] = GymEnv
    env = GymEnv(**env_desc["config"])

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    env_desc["possible_agents"] = env.possible_agents
    env.close()

    agent_mapping_func = lambda agent: "share"

    settings.USE_REMOTE_LOGGER = False
    exp_cfg = {
        "expr_group": "single_instance",
        "expr_name": "sample",
        "file_stream": False,
    }
    # ========================================================================================================

    global_step = 0

    # ==================================== Init rollout =========================================
    learners = {}
    for env_agent_id in env.possible_agents:
        interface_id = agent_mapping_func(env_agent_id)
        if learners.get(interface_id) is None:
            learners[interface_id] = get_training_agent(
                config["training"]["interface"]["type"]
            )(
                assign_id=interface_id,
                env_desc=env_desc,
                training_agent_mapping=None,
                algorithm_candidates=config["algorithms"],
                observation_spaces=observation_spaces,
                action_spaces=action_spaces,
                exp_cfg=exp_cfg,
            )
        learners[interface_id].register_env_agent(env_agent_id)

    rollout_handler = RolloutWorker(
        worker_index=None,
        env_desc=env_desc,
        metric_type=config["rollout"]["metric_type"],
        remote=False,
        exp_cfg=exp_cfg,
    )
    # =======================================================================================================

    # ================================= Register policies for agents and create buffers =====================
    trainable_policy_mapping = {}
    for learner in learners.values():
        for agent in learner.agent_group():
            pid, policy = learner.load_single_policy(agent, args.model_path)
            # record policy information
            # register policy into rollout handlers
            rollout_handler.update_population(agent, pid, policy)
            trainable_policy_mapping[agent] = pid

    agent_episodes = {
        aid: Episode(
            env_desc["config"]["env_id"],
            trainable_policy_mapping,
            capacity=args.sample_num,
        )
        for aid in env_desc["possible_agents"]
    }

    stationary_policy_distribution = {
        aid: {pid: 1.0} for aid, pid in trainable_policy_mapping.items()
    }
    # ======================================================================================================

    # ==================================== Main loop =======================================================
    total_frames = 0
    min_size = 0
    with Log.stat_feedback() as (statistic_seq, processed_statistics):
        while min_size < args.sample_num:
            statistics, num_frames = rollout_handler.sample(
                callback=rollout_func.simultaneous,
                num_episodes=args.parallel_num,
                fragment_length=config["rollout"]["fragment_length"],
                policy_combinations=[trainable_policy_mapping],
                explore=False,
                role="rollout",
                policy_distribution=stationary_policy_distribution,
                threaded=False,
                episodes=agent_episodes,
            )
            statistic_seq.extend(statistics)
            min_size = min([e.size for e in agent_episodes.values()])
            total_frames += num_frames

    print("rollout statistics:")
    pprint.pprint(processed_statistics[0])
    print("sampled new frames:", total_frames, "total_size:", min_size)

    save_dir = os.path.join(
        BASE_DIR,
        "demos",
        env_desc["config"]["env_id"],
        list(config["algorithms"].keys())[0],
    )
    os.makedirs(save_dir, exist_ok=True)

    single_agent_id = possible_agents[0]
    single_agent_episode = list(agent_episodes.values())[0]
    table_name = "{}_{}".format(env_desc["config"]["env_id"], single_agent_id)
    single_agent_table = Table(name=table_name, multi_agent=False)
    single_agent_table._episode = single_agent_episode
    # single_agent_table.set_episode(
    #     {single_agent_id: single_agent_episode}, single_agent_episode.capacity
    # )
    # single_agent_table.fill(**single_agent_episode.data)
    print(single_agent_table.size)
    single_agent_table.dump(save_dir, name=f"demo_{args.sample_num}")
