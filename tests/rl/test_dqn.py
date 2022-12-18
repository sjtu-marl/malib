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

import gym
import torch

from gym import spaces
from malib.rl.dqn import DQNPolicy, DQNTrainer


@pytest.mark.parametrize(
    "observation_space,action_space",
    [
        [
            spaces.Tuple([spaces.Box(low=-1.0, high=1.0, shape=(4,))]),
            spaces.Tuple([spaces.Discrete(4)]),
        ],
        [spaces.Box(low=-1.0, high=1.0, shape=(4,)), spaces.Discrete(4)],
    ],
)
@pytest.mark.parametrize("use_cuda", [False, True])
def test_dqn_policy(
    observation_space: gym.Space, action_space: gym.Space, use_cuda: bool
):
    model_config = {}
    custom_config = {"use_cuda": use_cuda}

    policy = DQNPolicy(observation_space, action_space, model_config, custom_config)
    policy.reset()
    policy.save("./dqn.pkl")
    policy.load("./dqn.pkl")

    observation = observation_space.sample()
    observation = torch.as_tensor(observation).to(
        dtype=torch.float32, device=policy.device
    )
    act_mask = None
    for evaluate in [True, False]:
        policy.compute_action(observation, act_mask, evaluate)
