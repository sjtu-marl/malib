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

from malib import rl
from malib.utils.tianshou_batch import Batch
from malib.utils.episode import Episode
from malib.mocker.mocker_utils import use_ray_env
from malib.rollout.envs.mdp import env_desc_gen
from malib.agent.indepdent_agent import IndependentAgent


def start_learner(env_id: str, algorithm: Any):
    experiment_tag = "test_"
    agent_mapping_func = lambda agent: agent
    env_desc = env_desc_gen(env_id=env_id)
    learners = {
        agent: IndependentAgent(
            experiment_tag=experiment_tag,
            runtime_id=agent,
            log_dir="./logs",
            env_desc=env_desc,
            algorithms={
                "default": (
                    algorithm.POLICY,
                    algorithm.TRAINER,
                    algorithm.DEFAULT_CONFIG["model_config"],
                    {},
                )
            },
            agent_mapping_func=agent_mapping_func,
            governed_agents=[agent],
            trainer_config=algorithm.DEFAULT_CONFIG["training_config"],
            custom_config={},
        )
        for agent in env_desc["possible_agents"]
    }
    for learner in learners.values():
        learner.connect(max_tries=2)
    return learners


@pytest.mark.parametrize("env_id", ["two_round_dmdp"])
@pytest.mark.parametrize("algorithm", [rl.pg, rl.a2c, rl.dqn])
class TestIndependentAgent:
    def test_policy_add(self, env_id, algorithm):
        with use_ray_env():
            learners = start_learner(env_id, algorithm)
            for learner in learners.values():
                learner.add_policies(n=1)

    def test_parameter_sync(self, env_id, algorithm):
        with use_ray_env():
            learners = start_learner(env_id, algorithm)
            for learner in learners.values():
                learner.add_policies(n=1)
                # then sync parameter to remote parameter server
                learner.push()
                # also pull down
                learner.pull()

    def test_multiagent_post_process(self, env_id, algorithm):
        with use_ray_env():
            learners = start_learner(env_id, algorithm)
            for learner in learners.values():
                batch = learner.multiagent_post_process((Batch(), None))
                assert isinstance(batch, Batch)
                with pytest.raises(TypeError):
                    learner.multiagent_post_process("fefefefe")

    def test_training_pipeline(self, env_id, algorithm):
        with use_ray_env():
            learners = start_learner(env_id, algorithm)
            for learner in learners.values():
                learner.add_policies(n=1)
