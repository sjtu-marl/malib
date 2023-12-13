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

from typing import Any

import pytest
import importlib
import gym
import numpy as np

from malib.utils.general import merge_dicts
from malib.rl.config import Algorithm
from malib.learner.learner import MAX_MESSAGE_LENGTH
from malib.learner.indepdent_learner import IndependentAgent
from malib.backend.dataset_server.data_loader import DynamicDataset


def construct_dataset(
    feature_handler=None, feature_handler_cls=None, feature_handler_kwargs=None
):
    return DynamicDataset(
        grpc_thread_num_workers=1,
        max_message_length=MAX_MESSAGE_LENGTH,
        feature_handler=feature_handler,
        feature_handler_cls=feature_handler_cls,
        **feature_handler_kwargs
    )


def construct_learner(
    obs_space,
    act_space,
    algorithm,
    governed_agents,
    custom_config=None,
    dataset=None,
    feature_handler_gen=None,
) -> IndependentAgent:
    return IndependentAgent(
        runtime_id=None,
        log_dir=None,
        observation_space=obs_space,
        action_space=act_space,
        algorithm=algorithm,
        governed_agents=governed_agents,
        custom_config=custom_config,
        dataset=dataset,
        feature_handler_gen=feature_handler_gen,
    )


def construct_algorithm(module_path, model_config={}, trainer_config={}):
    # import policy, trainer and default config from a given module
    policy_cls = importlib.import_module(module_path).Policy
    trainer_cls = importlib.import_module(module_path).Trainer
    default_config = importlib.import_module(module_path).DEFAULT_CONFIG

    return Algorithm(
        policy=policy_cls,
        trainer=trainer_cls,
        model_config=merge_dicts(default_config.MODEL_CONFIG, model_config),
        trainer_config=merge_dicts(default_config.TRAINING_CONFIG, trainer_config),
    )


from malib.mocker.mocker_utils import FakeFeatureHandler
from malib.rollout.episode import Episode


@pytest.mark.parametrize("module_path", [
    'malib.rl.random'
])
class TestIndependentAgent:
    def test_learner_with_outer_dataset(self, module_path):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)
        np_memory = {
            Episode.CUR_OBS: np.zeros()
        }
        governed_agents = ["default"]

        dataset = construct_dataset(
            feature_handler=FakeFeatureHandler(
                {
                    Episode.CUR_OBS: obs_space,
                    Episode.ACTION: act_space,
                },
                np_memory,
                block_size=100,
                device="cpu",
            )
        )
        algorithm = construct_algorithm(module_path)
        learner = construct_learner(
            algorithm, governed_agents, custom_config=None, dataset=dataset
        )

        for _ in range(10):
            learner.step(prints=True)

    def test_learner_with_outer_feature_handler(self):
        pass

    def test_learner_with_feature_handler_gen(self):
        pass

    def test_learner_with_dataset_gen(self):
        pass
