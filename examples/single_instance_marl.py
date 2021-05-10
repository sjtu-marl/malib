"""
Single instance run MPE with marl
"""

import argparse
import os
import pprint

import yaml
import importlib

import numpy as np

from malib import settings
from malib.runner import start_logger, terminate_logger
from malib.utils.typing import AgentID, PolicyID, Dict, Any


from malib.agent import get_training_agent
from malib.rollout.rollout_worker import RolloutWorker
from malib.rollout.rollout_func import rollout_wrapper

from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.logger import Log, get_logger


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(
    "Single instance of MARL training on mpe environments."
)
parser.add_argument(
    "--config", type=str, help="YAML configuration path.", required=True
)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    # ================== Clean configuration =================================================================
    env_desc = config["env_description"]
    env_desc["config"] = env_desc.get("config", {})
    # load creator
    env_module = importlib.import_module(f"pettingzoo.mpe.{env_desc['id']}")
    env_creator = (
        env_module.env
        if config["rollout"]["callback"] == "sequential"
        else env_module.parallel_env
    )
    env_desc["creator"] = env_creator
    env = env_creator(**env_desc["config"])

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    env_desc["possible_agents"] = env.possible_agents
    env.close()

    agent_mapping_func = lambda agent: "share"
    exp_cfg = {"expr_group": config["group"], "expr_name": config["name"]}
    # ========================================================================================================

    global_step = 0

    start_logger(exp_cfg)

    logger = get_logger(
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
        agent: Episode(
            env_desc["id"],
            policy_id=trainable_policy_mapping[agent],
            capacity=config["dataset_config"]["episode_capacity"],
            other_columns=["times"],
        )
        for agent in env.possible_agents
    }
    # ======================================================================================================

    # ==================================== Main loop =======================================================
    min_size = 0
    while min_size < config["dataset_config"]["learning_start"]:
        statistics, _ = rollout_handler.sample(
            callback=rollout_wrapper(
                agent_episodes, rollout_type=config["rollout"]["callback"]
            ),
            fragment_length=config["rollout"]["fragment_length"],
            behavior_policy_mapping=trainable_policy_mapping,
            num_episodes=[config["rollout"]["episode_seg"]],
            role="rollout",
            threaded=False,
        )
        min_size = min([e.size for e in agent_episodes.values()])

    for epoch in range(1000):
        print(f"==================== epoch #{epoch} ===============")
        _ = rollout_handler.sample(
            callback=rollout_wrapper(
                agent_episodes, rollout_type=config["rollout"]["callback"]
            ),
            fragment_length=config["rollout"]["fragment_length"],
            behavior_policy_mapping=trainable_policy_mapping,
            num_episodes=[config["rollout"]["episode_seg"]],
            role="rollout",
            threaded=False,
        )
        with Log.stat_feedback(
            log=True, logger=logger, worker_idx="Rollout", global_step=epoch
        ) as (statistic_seq, processed_statistics):
            statistics, _ = rollout_handler.sample(
                callback=rollout_wrapper(
                    None, rollout_type=config["evaluation"]["callback"]
                ),
                fragment_length=config["rollout"]["fragment_length"],
                behavior_policy_mapping=trainable_policy_mapping,
                num_episodes=[config["evaluation"]["num_episodes"]],
                role="rollout",
                threaded=False,
                explore=False,
            )
            statistic_seq.extend(statistics[0])

        print("-------------- rollout --------")
        pprint.pprint(processed_statistics[0])

        idxes = np.random.choice(min_size, config["training"]["config"]["batch_size"])
        batches = {}
        for agent in env.possible_agents:
            batch, _ = agent_episodes[agent].sample(idxes)
            batches[agent] = batch
        # check timestamp
        for i in range(config["training"]["config"]["batch_size"]):
            times = [b["times"][i] for b in batches.values()]
            assert max(times) == min(times), (max(times), min(times))

        print("-------------- traininig -------")
        for iid, interface in learners.items():
            with Log.stat_feedback(
                log=True, logger=logger, worker_idx=iid, global_step=global_step
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
    terminate_logger()
