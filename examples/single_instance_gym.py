"""
Single instance run Gym with marl
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
from malib.backend.datapool.offline_dataset_server import Episode


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(
    "Single instance of MARL training on gym environments."
)
parser.add_argument(
    "--config", type=str, help="YAML configuration path.", required=True
)
parser.add_argument(
    "--threaded", action="store_true", help="Whether use threaded sampling."
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

    if args.threaded:
        ray.init()

    exp_cfg = logger.start(group="single_instance", name="marl")
    # ========================================================================================================

    global_step = 0

    _logger = logger.get_logger(
        log_level=settings.LOG_LEVEL,
        name="single_instance_marl",
        remote=settings.USE_REMOTE_LOGGER,
        mongo=settings.USE_MONGO_LOGGER,
        **exp_cfg,
    )

    # ==================================== Init learners and rollout =========================================
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
            pid, policy = learner.add_policy_for_agent(agent, trainable=True)
            # record policy information
            # register policy into rollout handlers
            rollout_handler.update_population(agent, pid, policy)
            trainable_policy_mapping[agent] = pid

    agent_episodes = {
        aid: Episode(
            env_desc["config"]["env_id"],
            trainable_policy_mapping,
            capacity=config["dataset_config"]["episode_capacity"],
        )
        for aid in env_desc["possible_agents"]
    }

    stationary_policy_distribution = {
        aid: {pid: 1.0} for aid, pid in trainable_policy_mapping.items()
    }
    # ======================================================================================================

    # ==================================== Main loop =======================================================
    for epoch in range(1000):
        print(f"\n==================== epoch #{epoch} ===============")
        total_frames = 0
        min_size = 0
        with Log.stat_feedback(
            log=True,
            logger=_logger,
            worker_idx="Rollout",
            global_step=epoch,
            group="single_instance",
        ) as (statistic_seq, processed_statistics):
            while min_size < config["dataset_config"]["learning_start"]:
                statistics, num_frames = rollout_handler.sample(
                    callback=rollout_func.simultaneous,
                    num_episodes=config["rollout"]["num_episodes"],
                    fragment_length=config["rollout"]["fragment_length"],
                    policy_combinations=[trainable_policy_mapping],
                    explore=True,
                    role="rollout",
                    policy_distribution=stationary_policy_distribution,
                    threaded=args.threaded,
                    episodes=agent_episodes,
                )
                statistic_seq.extend(statistics)
                min_size = min([e.size for e in agent_episodes.values()])
                total_frames += num_frames

        print("rollout statistics:")
        pprint.pprint(processed_statistics[0])
        print("sampled new frames:", total_frames, "total_size:", min_size)

        batches = {}
        idxes = np.random.choice(min_size, config["training"]["config"]["batch_size"])
        for agent in env.possible_agents:
            batch = agent_episodes[agent].sample(idxes=idxes)
            batches[agent] = batch

        print("-------------- traininig -------")
        for iid, interface in learners.items():
            with Log.stat_feedback(
                log=True,
                logger=_logger,
                worker_idx=iid,
                global_step=global_step,
                group="single_instance",
            ) as (statistics_seq, processed_statistics):
                statistics_seq.append(
                    learners[iid].optimize(
                        policy_ids=trainable_policy_mapping.copy(),
                        batch=batches,
                        training_config=config["training"]["config"],
                    )
                )
            pprint.pprint(processed_statistics[0])
        global_step += 1
    # =====================================================================================================
    logger.terminate()
