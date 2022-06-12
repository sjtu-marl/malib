from argparse import ArgumentParser

import random
import time
import ray

from malib.utils.logging import Logger
from malib.utils.timing import Timing
from malib.remote.interface import RemoteInterface
from malib.algorithm.pg import PGPolicy
from malib.rollout.envs.env import Environment
from malib.rollout.envs.gym import env_desc_gen


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

    env_description = env_desc_gen(env_id=args.env_id, scenario_configs={})

    obs_spaces = env_description["observation_spaces"]
    act_spaces = env_description["action_spaces"]

    # policy = PGPolicy(observation_space=obs_spaces['agent'], action_space=act_spaces["agent"], model_config={}, custom_config={})
    policy = PolicyServer.as_remote().remote(
        observation_space=obs_spaces["agent"],
        action_space=act_spaces["agent"],
        model_config={},
        custom_config={},
    )
    preprocessor = ray.get(policy.get_preprocessor.remote())

    timer = Timing()

    try:
        env: Environment = env_description["creator"](**env_description["config"])
        cnt = 0
        Logger.info("start performance evaluation")
        start = time.time()

        while True:
            with timer.time_avg("reset"):
                raw_obs = env.reset()[0]["agent"]

            done = False
            while not done:
                with timer.time_avg("obs_transform"):
                    obs = preprocessor.transform(raw_obs)

                with timer.time_avg("action_compute"):
                    action, action_dist, logits, state = ray.get(
                        policy.compute_action.remote(
                            obs, action_mask=None, evaluate=random.choice([False, True])
                        )
                    )

                with timer.time_avg("env_step"):
                    raw_obs, act_mask, rew, done, info = env.step({"agent": action[0]})

                done = done["agent"]
                raw_obs = raw_obs["agent"]

                cnt += 1

    except KeyboardInterrupt as e:
        fps = cnt / (time.time() - start)
        Logger.warning(
            f"Keyboard interrupt detected, end evaluation. Average evaluation performance:\nFPS = {fps}\nAVG_TIMER={timer.todict()}"
        )
    finally:
        env.close()
