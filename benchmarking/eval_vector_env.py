from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import random
import time

import numpy as np
import ray

from ray.util.queue import Queue

from malib.utils.timing import Timing
from malib.utils.logging import Logger
from malib.remote.interface import RemoteInterface
from malib.rl.pg import PGPolicy
from malib.rollout.envs.env import Environment
from malib.rollout.envs.gym import env_desc_gen
from malib.rollout.envs.vector_env import VectorEnv
from malib.rollout.envs.async_vector_env import AsyncVectorEnv


def compute_action_loop(
    policy: PGPolicy, sender: Queue, recver: Queue, batch_size: int = 10
):
    while True:
        if recver.empty():
            continue

        buffer = []
        while not recver.empty() and len(buffer) < batch_size:
            buffer.append(recver.get_nowait())
        observation, action_mask, evaluate = list(zip(*buffer))
        observation = np.stack(observation).squeeze()
        if action_mask[0] is not None:
            action_mask = np.stack(action_mask).squeeze()
        evaluate = evaluate[0]
        action, action_dist, logits, state = policy.compute_action(
            observation, action_mask, evaluate
        )

        # combat buffer


class PolicyServer(RemoteInterface):
    def __init__(self, observation_space, action_space, model_config, custom_config):
        self.policy = PGPolicy(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )
        self.sender: Queue = None
        self.recver: Queue = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

    def connect(self):
        self.sender = Queue(actor_options={"num_cpus": 0})
        self.recver = Queue(actor_options={"num_cpus": 0})
        self.thread_pool.submit(compute_action_loop, self.sender, self.recver)
        return {"sender": self.recver, "recver": self.sender}

    def compute_action(self, observation, action_mask, evaluate):
        return self.policy.compute_action(
            observation, action_mask=action_mask, evaluate=evaluate
        )

    def get_preprocessor(self):
        return self.policy.preprocessor


if __name__ == "__main__":
    parser = ArgumentParser("Multi-agent reinforcement learning.")
    parser.add_argument("--log_dir", default="./logs/", help="Log directory.")
    parser.add_argument("--env_id", default="CartPole-v1", help="gym environment id.")
    parser.add_argument(
        "--remote_actor", action="store_true", help="enable remote actor or not."
    )
    parser.add_argument("--num_envs", default=10, type=int, help="enrionment number.")

    args = parser.parse_args()

    env_desc = env_desc_gen(env_id=args.env_id, scenario_configs={})

    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]
    env_cls = env_desc["creator"]
    env_config = env_desc["config"]

    timer = Timing()
    timer.clear()

    try:
        env = AsyncVectorEnv(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            creator=env_cls,
            configs=env_config,
            preset_num_envs=args.num_envs,
        )

        if args.remote_actor:
            policy = PolicyServer.as_remote().remote(
                observation_space=obs_spaces["agent"],
                action_space=act_spaces["agent"],
                model_config={},
                custom_config={},
            )
            preprocessor = ray.get(policy.get_preprocessor.remote())
        else:
            policy = PGPolicy(
                observation_space=obs_spaces["agent"],
                action_space=act_spaces["agent"],
                model_config={},
                custom_config={},
            )
            preprocessor = policy.preprocessor

        cnt = 0
        Logger.info("start performance evaluation")
        start = time.time()

        while True:
            with timer.time_avg("reset"):
                env_rets = env.reset(fragment_length=2000, max_step=200)

            done = False
            while not env.is_terminated():
                with timer.time_avg("obs_transform"):
                    obs_list = []
                    env_ids = []
                    for env_id, env_ret in env_rets.items():
                        env_ids.append(env_id)
                        raw_obs = env_ret[0]["agent"]
                        obs_list.append(raw_obs)
                    obs_list = preprocessor.transform(obs_list)

                with timer.time_avg("action_compute"):
                    if args.remote_actor:
                        action, action_dist, logits, state = ray.get(
                            policy.compute_action.remote(
                                obs_list,
                                action_mask=None,
                                evaluate=random.choice([False, True]),
                            )
                        )
                    else:
                        action, action_dist, logits, state = policy.compute_action(
                            obs_list,
                            action_mask=None,
                            evaluate=random.choice([False, True]),
                        )
                    # recover to environment actions
                    env_actions = {}
                    # print(env_ids, action)
                    for env_id, _action in zip(env_ids, action):
                        env_actions[env_id] = {"agent": _action}

                with timer.time_avg("env_step"):
                    env_rets = env.step(env_actions)

                cnt += 1
                if cnt % 1000 == 0:
                    fps = cnt / (time.time() - start)
                    Logger.info(
                        f"Average evaluation performance:\nFPS = {fps}\nAVG_TIMER={timer.todict()}"
                    )

    except KeyboardInterrupt as e:
        fps = cnt / (time.time() - start)
        Logger.warning(
            f"Keyboard interrupt detected, end evaluation. Average evaluation performance:\nFPS = {fps}\nAVG_TIMER={timer.todict()}"
        )
    finally:
        env.close()
