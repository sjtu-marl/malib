from argparse import ArgumentParser

import random
import time

import numpy as np
import ray

from malib.utils.timing import Timing
from malib.utils.logging import Logger
from malib.remote.interface import RemoteInterface
from malib.algorithm.pg import PGPolicy
from malib.rollout.envs.env import Environment
from malib.rollout.envs.gym import env_desc_gen
from malib.rollout.envs.vector_env import VectorEnv
from malib.rollout.envs.async_vector_env import AsyncVectorEnv


class PolicyServer(RemoteInterface):
    def __init__(self, observation_space, action_space, model_config, custom_config):
        self.policy = PGPolicy(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

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
            preset_num_envs=10,
        )

        # policy = PGPolicy(observation_space=obs_spaces['agent'], action_space=act_spaces["agent"], model_config={}, custom_config={})
        policy = PolicyServer.as_remote().remote(
            observation_space=obs_spaces["agent"],
            action_space=act_spaces["agent"],
            model_config={},
            custom_config={},
        )
        preprocessor = ray.get(policy.get_preprocessor.remote())

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
                        obs_list.append(preprocessor.transform(raw_obs))
                    obs_list = np.stack(obs_list).squeeze()

                with timer.time_avg("action_compute"):
                    action, action_dist, logits, state = ray.get(
                        policy.compute_action.remote(
                            obs_list,
                            action_mask=None,
                            evaluate=random.choice([False, True]),
                        )
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
