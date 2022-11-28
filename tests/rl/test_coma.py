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

from typing import Dict, List, Any, Tuple
from functools import partial

import pytest
import gym
import numpy as np
import torch

from gym import spaces

from malib.utils.episode import Episode
from malib.utils.tianshou_batch import Batch
from malib.rl.coma.critic import COMADiscreteCritic
from malib.rl.coma.trainer import COMATrainer


def gen_agent_batch(
    agents: List[str], use_timestep: bool, device: str
) -> Dict[str, Batch]:
    batches = {}
    batch_size = 64
    time_step = 100

    for agent in agents:
        if not use_timestep:
            batch = Batch(
                {
                    Episode.CUR_STATE: np.random.random((batch_size, 12)),
                    Episode.CUR_OBS: np.random.random((batch_size, 3)),
                    Episode.ACTION: np.random.random((batch_size, 3)),
                }
            )
        else:
            batch = Batch(
                {
                    Episode.CUR_STATE: np.random.random((batch_size, time_step, 12)),
                    Episode.CUR_OBS: np.random.random((batch_size, time_step, 3)),
                    Episode.ACTION: np.random.random((batch_size, time_step, 3)),
                }
            )
        batch.to_torch(device=device)
        batches[agent] = batch

    centralized_obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(12 + 3 + 3 * len(agents),)
    )

    return batches, centralized_obs_space, batch_size, time_step


@pytest.mark.parametrize(
    "net_type,kwargs",
    [
        [None, {"hidden_sizes": [64, 64], "activation": "ReLU", "action_shape": 1}],
        ["mlp", {"hidden_sizes": [64, 64]}],
        ["general_net", {"hidden_sizes": [64, 64]}],
        ["rnn", {"layer_num": 2}],
    ],
)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_timestep", [False, True])
def test_centralized_critic(
    net_type: str, kwargs: Dict[str, Any], device: str, use_timestep: bool
):
    if not torch.cuda.is_available():
        device = "cpu"
    agent_batch, centralized_obs_space, batch_size, time_step = gen_agent_batch(
        [f"agent_{i}" for i in range(3)], use_timestep, device
    )
    critic = COMADiscreteCritic(centralized_obs_space, net_type, device, **kwargs)
    critic.to(device)

    values = critic(agent_batch)
    if isinstance(values, Tuple):
        values = values[0]

    if use_timestep:
        expected_shape = (batch_size, time_step, 3, 1)
        assert (
            values.shape == expected_shape
        ), f"net_type: {net_type}, expected_shape: {expected_shape}, valueshape: {values.shape}"
    else:
        expected_shape = (batch_size, 3, 1)
        assert (
            values.shape == expected_shape
        ), f"net_type: {net_type}, expected_shape: {expected_shape}, valueshape: {values.shape}"

    if "cuda" in device:
        assert "cuda" in values.device.type
    else:
        assert "cpu" in values.device.type


# def test_coma_training(
#     net_type: str, kwargs: Dict[str, Any], device: str, use_timestep: bool
# ):
#     agent_batch, centralized_obs_space, batch_size, time_step = gen_agent_batch(
#         [f"agent_{i}" for i in range(3)], use_timestep, device
#     )
#     critic_creator = partial(
#         COMADiscreteCritic, centralized_obs_space, net_type, device, **kwargs
#     )
#     trainer = COMATrainer(training_config, critic_creator, policy_instance)
#     loss = trainer(agent_batch)
