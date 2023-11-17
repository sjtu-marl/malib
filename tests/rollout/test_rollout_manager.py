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

from typing import Dict, Any, Callable

import pytest
import ray

from gym import spaces
from pytest_mock import MockerFixture

from malib.common.task import RolloutTask
from malib.common.strategy_spec import StrategySpec
from malib.rollout.config import RolloutConfig
from malib.rollout.manager import RolloutWorkerManager
from malib.rl.random import RandomPolicy
from malib.scenarios.scenario import form_group_info
from malib.learner.manager import LearnerManager
from malib.learner.config import LearnerConfig
from malib.rollout.inference.manager import InferenceManager
from test_pb_rollout_worker import (
    feature_handler_meta_gen,
    FakeFeatureHandler,
    FakeLearner,
    gen_common_requirements,
)


def create_manager(
    stopping_conditions: Dict[str, Any],
    rollout_config: Dict[str, Any],
    env_desc: Dict[str, Any],
    agent_mapping_func: Callable,
):
    manager = RolloutWorkerManager(
        stopping_conditions=stopping_conditions,
        num_worker=1,
        group_info=form_group_info(env_desc, agent_mapping_func),
        rollout_config=rollout_config,
        env_desc=env_desc,
        log_dir="./logs",
    )
    return manager


@pytest.mark.parametrize("n_players", [1, 2])
class TestRolloutManager:
    def test_rollout_task_send(self, mocker: MockerFixture, n_players: int):
        with ray.init(local_mode=True):
            env_desc, algorithm, rollout_config, group_info = gen_common_requirements(
                n_players
            )
            inference_namespace = "test_pb_rolloutworker"
            manager = create_manager(
                stopping_conditions={"rollout": {"max_iteration": 2}},
                rollout_config=RolloutConfig(),
                env_desc=env_desc,
                agent_mapping_func=lambda agent: "default",
            )

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
                log_dir="./logs",
            )

            infer_manager = InferenceManager(
                group_info=group_info,
                ray_actor_namespace=inference_namespace,
                algorithm=algorithm,
                model_entry_point=learner_manager.learner_entrypoints,
            )

            rollout_config.inference_entry_points = infer_manager.inference_entry_points

            strategy_specs = {
                agent: StrategySpec(
                    policy_cls=RandomPolicy,
                    observation_space=env_desc["observation_spaces"][agent],
                    action_space=env_desc["action_spaces"][agent],
                    policy_ids=["policy_0"],
                )
                for agent in env_desc["possible_agents"]
            }

            task = RolloutTask(
                strategy_specs=strategy_specs,
                stopping_conditions={"max_iteration": 10},
                data_entrypoints=None,
            )

            results = manager.submit(task, wait=True)
