# -*- coding: utf-8 -*-
import argparse
import datetime
import os

# from pettingzoo.classic import leduc_holdem_v2 as leduc_holdem
# https://www.pettingzoo.ml/classic/leduc_holdem
from collections import defaultdict

import numpy as np

from malib import settings
from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.poker import poker_aec_env as leduc_holdem
from malib.runner import run
from malib.utils.metrics import get_metric

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser("PSRO training on mpe environments.")

parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--fragment_length", type=int, default=10240)
parser.add_argument("--worker_num", type=int, default=3)
parser.add_argument("--log_dir", type=str, default=settings.LOG_DIR)
parser.add_argument("--algorithm", type=str, default="DQN")

args = parser.parse_args()


num_total_training_episode = 55

args.algorithm = "DQN"
# args.algorithm = "SAC"
args.batch_size = 128
args.num_epoch = 8
args.fragment_length = 10
num_episode = 1000

# TODO(ming): add to offline dataset
off_policy_start_sample_timestamp = 500
buffer_size = 200000
num_simulation = 100

now_time = now = datetime.datetime.now(
    tz=datetime.timezone(datetime.timedelta(hours=8))
)
args.log_dir = os.path.join(
    # settings.LOG_DIR, f"DQN/{now_time.strftime('%Y%m%d_%H_%M_%S')}"
    settings.LOG_DIR,
    f"SAC/{now_time.strftime('%Y%m%d_%H_%M_%S')}",
)
args.work_num = 3


def psro_rollout_func(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
):
    env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))
    # for _ in range(num_episode):
    env.reset()

    # metric.add_episode(f"simulation_{policy_combination_mapping}")
    metric = get_metric(metric_type)(
        env.possible_agents if trainable_pairs is None else list(trainable_pairs.keys())
    )
    if behavior_policy_mapping is None:
        for agent in agent_interfaces.values():
            agent.reset()
    behavior_policy_mapping = behavior_policy_mapping or {
        _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
    }
    agent_episode = {
        agent: Episode(
            env_desc["id"],
            behavior_policy_mapping[agent],
            other_columns=["next_action_mask"],
            capacity=max_iter,
        )
        for agent in (trainable_pairs or env.possible_agents)
    }

    (
        observations,
        actions,
        action_dists,
        next_observations,
        rewards,
        dones,
        infos,
        next_action_mask,
    ) = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    for aid in env.agent_iter(max_iter=max_iter):
        observation, reward, done, info = env.last()
        if isinstance(observation, dict):
            info = {"action_mask": observation["action_mask"]}
        action_mask = observation["action_mask"]
        observation = agent_interfaces[aid].transform_observation(
            observation, behavior_policy_mapping[aid]
        )
        observations[aid].append(observation)
        rewards[aid].append(reward)
        dones[aid].append(done)
        next_action_mask[aid].append(action_mask)
        info["policy_id"] = behavior_policy_mapping[aid]
        if not done:
            action, action_probs, extra_info = agent_interfaces[aid].compute_action(
                observation, **info
            )
            actions[aid].append(action)
            action_dists[aid].append(action_probs)
        else:
            action = None
        env.step(action)
        metric.step(
            aid,
            behavior_policy_mapping[aid],
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )

    # metric.end()
    for k in agent_episode:
        obs = observations[k]
        cur_len = len(obs)
        agent_episode[k].fill(
            **{
                Episode.CUR_OBS: np.stack(obs[: cur_len - 1]),
                Episode.NEXT_OBS: np.stack(obs[1:cur_len]),
                Episode.DONES: np.stack(dones[k][1:cur_len]),
                Episode.REWARDS: np.stack(rewards[k][1:cur_len]),
                Episode.ACTIONS: np.stack(actions[k][: cur_len - 1]),
                Episode.ACTION_DIST: np.stack(action_dists[k][: cur_len - 1]),
                "next_action_mask": np.stack(next_action_mask[k][1:cur_len]),
            }
        )

    return metric.parse(), agent_episode


if __name__ == "__main__":
    env_config = {"fixed_player": True}

    env = leduc_holdem.env(**env_config)
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    run(
        group="psro",
        name="leduc_poker",
        env_description={
            "creator": leduc_holdem.env,
            "config": env_config,
            "id": "leduc_holdem",
            "possible_agents": possible_agents,
        },
        training={
            "interface": {
                "type": "independent",
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
            },
            "config": {
                "batch_size": args.batch_size,
                "update_interval": args.num_epoch,
            },
        },
        algorithms={
            args.algorithm: {
                "name": args.algorithm,
                "custom_config": {
                    "gamma": 1.0,
                    "eps_min": 0,
                    "eps_max": 1.0,
                    "eps_decay": 100,
                    "lr": 1e-2,
                },
            }
        },
        rollout={
            "type": "async",
            "stopper": "simple_rollout",
            "stopper_config": {"max_step": num_total_training_episode},
            "metric_type": "simple",
            "fragment_length": args.fragment_length,
            "num_episodes": num_episode,
            "episode_seg": 100,
            "callback": psro_rollout_func,
        },
        evaluation={
            "max_episode_length": 5,
            "num_episode": num_simulation,
        },  # dict(num_simulation=num_simulation, sim_max_episode_length=5),
        global_evaluator={
            "name": "psro",
            "config": {
                "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
            },
        },
        dataset_config={"episode_capacity": buffer_size},
    )
