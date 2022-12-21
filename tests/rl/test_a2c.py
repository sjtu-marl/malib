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

from functools import partial

import pytest

import gym
import numpy as np
import torch

from gym import spaces

from malib.utils.tianshou_batch import Batch
from malib.rl.a2c import A2CPolicy, A2CTrainer, DEFAULT_CONFIG


@pytest.mark.parametrize(
    "action_space",
    [
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Discrete(4),
        spaces.MultiBinary(4),
    ],
)
def test_a2c_policy(action_space: spaces.Space):
    """test different action space for the initialization of policy"""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    policy_func = partial(
        A2CPolicy, observation_space, action_space, DEFAULT_CONFIG["model_config"], {}
    )
    if not isinstance(action_space, (spaces.Box, spaces.Discrete)):
        with pytest.raises((TypeError, NotImplementedError)):
            policy_func()
    else:
        policy: A2CPolicy = policy_func()
        # call value function
        observation = observation_space.sample()
        observation = torch.as_tensor(observation).float().reshape(1, -1)
        policy.value_function(observation, False)


def test_a2c_trainer():
    """test reward norm and LR_scheduler"""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    policy = A2CPolicy(
        observation_space, action_space, DEFAULT_CONFIG["model_config"], {}
    )

    # create a fake batch
    training_config = DEFAULT_CONFIG["training_config"].copy()
    training_config["reward_norm"] = 1e-4
    trainer = A2CTrainer(training_config, policy_instance=policy)

    batch_size = 64
    obs_batch = np.random.random((batch_size,) + observation_space.shape)
    rew_batch = np.random.random((batch_size,))
    done_batch = np.zeros(batch_size).astype(np.float32)
    done_batch[-1] = 1.0
    act_batch = np.random.choice(action_space.n, batch_size)

    batch = Batch(
        obs=obs_batch, act=act_batch, obs_next=obs_batch, rew=rew_batch, done=done_batch
    )
    trainer(batch)
