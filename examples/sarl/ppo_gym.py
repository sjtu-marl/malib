import os
import time

from argparse import ArgumentParser

from gym import spaces

import numpy as np

from malib.utils.episode import Episode
from malib.learner import IndependentAgent
from malib.scenarios import sarl_scenario
from malib.rl.config import Algorithm
from malib.rl.ppo import PPOPolicy, PPOTrainer, DEFAULT_CONFIG
from malib.learner.config import LearnerConfig
from malib.rollout.config import RolloutConfig
from malib.rollout.envs.gym import env_desc_gen
from malib.backend.dataset_server.feature import BaseFeature


class FeatureHandler(BaseFeature):
    pass


def feature_handler_meta_gen(env_desc, agent_id):
    def f(device):
        # define the data schema
        _spaces = {
            Episode.DONE: spaces.Discrete(1),
            Episode.CUR_OBS: env_desc["observation_spaces"][agent_id],
            Episode.ACTION: env_desc["action_spaces"][agent_id],
            Episode.REWARD: spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            Episode.NEXT_OBS: env_desc["observation_spaces"][agent_id],
        }

        # you should know the maximum of replaybuffer before training
        np_memory = {
            k: np.zeros((100,) + v.shape, dtype=v.dtype) for k, v in _spaces.items()
        }

        return FeatureHandler(_spaces, np_memory, device)

    return f


if __name__ == "__main__":
    parser = ArgumentParser("Use PPO solve Gym tasks.")
    parser.add_argument("--log-dir", default="./logs/", help="Log directory.")
    parser.add_argument("--env-id", default="CartPole-v1", help="gym environment id.")
    parser.add_argument("--use-cuda", action="store_true")

    args = parser.parse_args()

    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    trainer_config["total_timesteps"] = int(1e6)
    trainer_config["use_cuda"] = args.use_cuda

    runtime_logdir = os.path.join(
        args.log_dir, f"gym/{args.env_id}/independent_ppo/{time.time()}"
    )

    if not os.path.exists(runtime_logdir):
        os.makedirs(runtime_logdir)

    scenario = sarl_scenario.SARLScenario(
        name=f"ppo-gym-{args.env_id}",
        log_dir=runtime_logdir,
        env_desc=env_desc_gen(env_id=args.env_id),
        algorithm=Algorithm(
            trainer=PPOTrainer,
            policy=PPOPolicy,
            model_config=None,  # use default
            trainer_config=trainer_config,
        ),
        learner_config=LearnerConfig(
            learner_type=IndependentAgent,
            feature_handler_meta_gen=feature_handler_meta_gen,
            custom_config={},
        ),
        rollout_config=RolloutConfig(
            num_workers=1,
        ),
        agent_mapping_func=lambda agent: agent,
        stopping_conditions={
            "training": {"max_iteration": int(1e10)},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )

    results = sarl_scenario.execution_plan(scenario=scenario, verbose=True)

    print(results)
