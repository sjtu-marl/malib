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
import numpy as np

from gym import spaces

from malib.rl.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG
from malib.utils.tianshou_batch import Batch


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
    if not torch.cuda.is_available():
        use_cuda = False
    custom_config = {"use_cuda": use_cuda}

    policy = DQNPolicy(observation_space, action_space, model_config, custom_config)
    policy.reset()
    policy.save("./dqn.pkl")
    policy.load("./dqn.pkl")

    observation = observation_space.sample()
    observation = torch.as_tensor(observation).to(
        dtype=torch.float32, device=policy.device
    )

    if isinstance(observation_space, spaces.Box):
        observation = observation.unsqueeze(0)

    if isinstance(action_space, spaces.Tuple):
        act_mask = np.ones((action_space.spaces[0].n))
    else:
        act_mask = np.ones(action_space.n)

    act_mask[1] = 0
    act_mask[3] = 0

    act_mask = torch.as_tensor(act_mask).to(dtype=torch.float32, device=policy.device)
    act_mask = act_mask.unsqueeze(0)

    for evaluate in [True, False]:
        policy.compute_action(observation, act_mask, evaluate)


def test_dqn_trainer():

    num_agents = 4
    batch_size = 64

    # post process, for the agent_dimension > 0
    observation_space = spaces.Tuple(
        [spaces.Box(low=-1, high=1, shape=(4,)) for _ in range(num_agents)]
    )
    action_space = spaces.Tuple([spaces.Discrete(3) for _ in range(num_agents)])

    policy = DQNPolicy(
        observation_space, action_space, model_config={}, custom_config={}
    )

    trainer = DQNTrainer(
        training_config=DEFAULT_CONFIG["training_config"], policy_instance=policy
    )
    trainer.reset()

    obs_batch = np.random.random((batch_size, num_agents, 4))
    rew_batch = np.random.random((batch_size, num_agents))
    done_batch = np.random.choice(2, batch_size).astype(np.float32)
    done_batch = np.tile(done_batch, reps=(1, num_agents))
    act_batch = np.random.choice(action_space.spaces[0].n, batch_size * num_agents)
    act_mask_batch = np.zeros((batch_size * num_agents, 3))
    act_mask_batch[np.arange(batch_size * num_agents), act_batch] = 1
    act_mask_batch = 1 - act_mask_batch

    act_batch = act_batch.reshape((batch_size, num_agents))
    act_mask_batch = act_mask_batch.reshape((batch_size, num_agents, 3))

    loss = trainer(
        Batch(
            obs=obs_batch,
            obs_next=obs_batch,
            act_mask_next=act_mask_batch,
            rew=rew_batch,
            done=done_batch,
            act=act_batch,
        )
    )
