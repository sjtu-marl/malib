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

"""for api test"""

import os
import pytest
import numpy as np

from malib import rl
from malib.rl.common import policy, trainer
from malib.utils.episode import Episode
from malib.utils.tianshou_batch import Batch
from malib.rollout.envs.mdp.env import MDPEnvironment


@pytest.mark.parametrize("algorithm", [rl.pg, rl.a2c, rl.dqn])
@pytest.mark.parametrize(
    "mdp_env_id",
    [
        "one_round_dmdp",
        "two_round_dmdp",
        "one_round_nmdp",
        "two_round_nmdp",
        "multi_round_nmdp",
    ],
)
class TestAlgorithm:
    @pytest.mark.parametrize("evaluation_mode", [False, True])
    def test_policy_construct(self, algorithm, mdp_env_id, evaluation_mode):
        env = MDPEnvironment(env_id=mdp_env_id)
        agent: policy.Policy = algorithm.POLICY(
            env.observation_spaces["agent"],
            env.action_spaces["agent"],
            algorithm.DEFAULT_CONFIG["model_config"],
            algorithm.DEFAULT_CONFIG["custom_config"],
        )

        done = False
        _, obs = env.reset()
        total_rew = 0
        cnt = 0
        while not done:
            obs = {k: agent.preprocessor.transform(v) for k, v in obs.items()}
            actions = {
                k: agent.compute_action(
                    v.reshape(1, -1), act_mask=None, evaluate=evaluation_mode
                )[0][0]
                for k, v in obs.items()
            }
            _, obs, rew, done, info = env.step(actions)
            done = done["__all__"]
            cnt += 1
            total_rew += rew["agent"]
        print(total_rew, cnt)

    def test_trainer_construct(self, algorithm, mdp_env_id):
        """Test for checking the interface calling of trainer, not guarantee for the correctness yet."""

        env = MDPEnvironment(env_id=mdp_env_id)
        policy: policy.Policy = algorithm.POLICY(
            env.observation_spaces["agent"],
            env.action_spaces["agent"],
            algorithm.DEFAULT_CONFIG["model_config"],
            algorithm.DEFAULT_CONFIG["custom_config"],
        )
        trainer: trainer.Trainer = algorithm.TRAINER(
            algorithm.DEFAULT_CONFIG["training_config"], policy_instance=policy
        )

        # reset for none
        trainer.reset()

        # reset with policy
        trainer.reset(policy_instance=policy)

        # reset with configuration
        trainer.reset(configs=algorithm.DEFAULT_CONFIG["training_config"])

        n_samples = 10

        raw_obs = [env.observation_spaces["agent"].sample() for _ in range(n_samples)]
        actions = np.asarray(
            [env.action_spaces["agent"].sample() for _ in range(n_samples)]
        )
        rewards = np.random.sample(n_samples)
        dones = np.random.choice([True, False], 10)
        obs = np.stack([policy.preprocessor.transform(x) for x in raw_obs])

        buffer = Batch(
            {
                Episode.CUR_OBS: obs,
                Episode.ACTION: actions,
                Episode.REWARD: rewards,
                Episode.DONE: dones,
                Episode.NEXT_OBS: np.roll(obs, -1),
            }
        )
        trainer(buffer)
