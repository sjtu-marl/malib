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

import pytest
import ray

from malib.common.task import RolloutTask
from malib.common.strategy_spec import StrategySpec
from malib.rl.random import RandomPolicy
from malib.rl.random import RandomTrainer
from malib.rl.config import Algorithm
from malib.rollout.envs.random import env_desc_gen
from malib.rollout.config import RolloutConfig
from malib.rollout.pb_rolloutworker import PBRolloutWorker
from malib.rollout.inference.manager import InferenceManager
from malib.scenarios.scenario import form_group_info


def gen_rollout_config(inference_server_type: str):
    return {
        "fragment_length": 100,
        "max_step": 10,
        "num_eval_episodes": 2,
        "num_threads": 1,
        "num_env_per_thread": 1,
        "num_eval_threads": 1,
        "use_subproc_env": False,
        "batch_mode": "timestep",
        "postprocessor_types": None,
        "eval_interval": 1,
        "inference_server": inference_server_type,
    }


def gen_common_requirements(n_player: int):
    env_desc = env_desc_gen(num_agents=n_player)

    algorithm = Algorithm(
        policy=RandomPolicy, trainer=RandomTrainer, model_config=None, trainer_config={}
    )

    rollout_config = RolloutConfig(
        num_workers=1,
        eval_interval=1,
        n_envs_per_worker=10,
        use_subproc_env=False,
        timelimit=256,
    )

    group_info = form_group_info(env_desc, lambda agent: "default")

    return env_desc, algorithm, rollout_config, group_info


import numpy as np

from gym import spaces
from malib.learner.manager import LearnerManager
from malib.learner.config import LearnerConfig
from malib.rollout.episode import Episode
from malib.mocker.mocker_utils import FakeLearner, FakeFeatureHandler


def feature_handler_meta_gen(env_desc, agent_id):
    def f(device):
        _spaces = {
            Episode.DONE: spaces.Discrete(1),
            Episode.CUR_OBS: env_desc["observation_spaces"][agent_id],
            Episode.ACTION: env_desc["action_spaces"][agent_id],
            Episode.REWARD: spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            Episode.NEXT_OBS: env_desc["observation_spaces"][agent_id],
        }
        np_memory = {
            k: np.zeros((1000,) + v.shape, dtype=v.dtype) for k, v in _spaces.items()
        }
        return FakeFeatureHandler(_spaces, np_memory, device=device)

    return f


@pytest.mark.parametrize("n_player", [1, 2])
class TestRolloutWorker:
    def test_rollout(self, n_player: int):
        with ray.init(local_mode=True):
            env_desc, algorithm, rollout_config, group_info = gen_common_requirements(
                n_player
            )

            obs_spaces = env_desc["observation_spaces"]
            act_spaces = env_desc["action_spaces"]
            agents = env_desc["possible_agents"]
            log_dir = "./logs"

            inference_namespace = "test_pb_rolloutworker"

            infer_manager = InferenceManager(
                group_info=group_info,
                ray_actor_namespace=inference_namespace,
                algorithm=algorithm,
                model_entry_point=None,
            )

            rollout_config.inference_entry_points = infer_manager.inference_entry_points

            strategy_specs = {
                agent: StrategySpec(
                    policy_cls=algorithm.policy,
                    observation_space=obs_spaces[agent],
                    action_space=act_spaces[agent],
                    identifier=agent,
                    model_config=algorithm.model_config,
                    policy_ids=["policy-0"],
                )
                for agent in agents
            }

            worker = PBRolloutWorker(
                env_desc=env_desc,
                agent_groups=group_info["agent_groups"],
                rollout_config=rollout_config,
                log_dir=log_dir,
            )

            task = RolloutTask(
                strategy_specs=strategy_specs,
                stopping_conditions={"max_iteration": 10},
                data_entrypoints=None,  # no data collect
            )
            stats = worker.rollout(task)

    def test_rollout_with_data_entrypoint(self, n_player: int):
        with ray.init():
            env_desc, algorithm, rollout_config, group_info = gen_common_requirements(
                n_player
            )

            obs_spaces = env_desc["observation_spaces"]
            act_spaces = env_desc["action_spaces"]
            agents = env_desc["possible_agents"]
            log_dir = "./logs"

            learner_manager = LearnerManager(
                stopping_conditions={"max_iteration": 10},
                algorithm=algorithm,
                env_desc=env_desc,
                agent_mapping_func=lambda agent: "default",
                group_info=group_info,
                learner_config=LearnerConfig(
                    learner_type=FakeLearner,
                    feature_handler_meta_gen=feature_handler_meta_gen,
                    custom_config=None,
                ),
                log_dir=log_dir,
            )
            # create a batch of inference servers, serve for rollout workers (shared among them)
            infer_manager = InferenceManager(
                group_info=group_info,
                algorithm=algorithm,
                model_entry_point=learner_manager.learner_entrypoints,
            )

            rollout_config.inference_entry_points = infer_manager.inference_entry_points
            assert (
                "default" in rollout_config.inference_entry_points
            ), rollout_config.inference_entry_points

            strategy_spaces = {
                agent: StrategySpec(
                    policy_cls=algorithm.policy,
                    observation_space=obs_spaces[agent],
                    action_space=act_spaces[agent],
                    identifier=agent,
                    model_config=algorithm.model_config,
                    policy_ids=["policy-0"],
                )
                for agent in agents
            }
            # create a single PB rollout worker, for task execution
            worker = PBRolloutWorker(
                env_desc=env_desc,
                agent_groups=group_info["agent_groups"],
                rollout_config=rollout_config,
                log_dir=log_dir,
            )

            print("PBRollout worker is ready to work!!!")

            task = RolloutTask(
                strategy_specs=strategy_spaces,
                stopping_conditions={"max_iteration": 10},
                data_entrypoints=learner_manager.data_entrypoints,
            )

            stats = worker.rollout(task)
        ray.shutdown()
