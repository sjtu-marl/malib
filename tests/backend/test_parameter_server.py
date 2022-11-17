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
import numpy as np
import torch

from gym import spaces

from malib import rl
from malib.backend.parameter_server import Table, ParameterServer
from malib.rl.common.policy import Policy
from malib.common.strategy_spec import StrategySpec


@pytest.mark.parametrize("optim_config", [None, {"type": "Adam", "lr": 1e-4}])
@pytest.mark.parametrize(
    "policy_cls,rl_default_config",
    [
        [rl.a2c.A2CPolicy, rl.a2c.DEFAULT_CONFIG],
        [rl.dqn.DQNPolicy, rl.dqn.DEFAULT_CONFIG],
        [rl.pg.PGPolicy, rl.pg.DEFAULT_CONFIG],
    ],
)
def test_parameter_table(optim_config, policy_cls, rl_default_config):
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3))
    action_space = spaces.Discrete(5)

    policy_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "model_config": rl_default_config["model_config"],
        "custom_config": rl_default_config["custom_config"],
        "kwargs": {},
    }
    policy_copy: Policy = policy_cls(
        observation_space=observation_space,
        action_space=action_space,
        model_config=rl_default_config["model_config"],
        custom_config=rl_default_config["custom_config"],
    )
    table = Table(
        policy_meta_data={
            "policy_cls": policy_cls,
            "optim_config": optim_config,
            "kwargs": policy_kwargs,
        }
    )

    # set weights from policy
    table.set_weights(policy_copy.state_dict())

    # check weights
    table_weights = table.get_weights()
    for k, v in policy_copy.state_dict().items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                assert torch.all(_v == table_weights[k][_k]), (k, _k)

    # TODO(ming): test gradient apply here, if the method has been implemented


@pytest.mark.parametrize("optim_config", [None, {"type": "Adam", "lr": 1e-4}])
@pytest.mark.parametrize(
    "policy_cls,rl_default_config",
    [
        [rl.a2c.A2CPolicy, rl.a2c.DEFAULT_CONFIG],
        [rl.dqn.DQNPolicy, rl.dqn.DEFAULT_CONFIG],
        [rl.pg.PGPolicy, rl.pg.DEFAULT_CONFIG],
    ],
)
def test_parameter_server(optim_config, policy_cls, rl_default_config):
    server = ParameterServer()

    server.start()

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3))
    action_space = spaces.Discrete(5)

    policy_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "model_config": rl_default_config["model_config"],
        "custom_config": rl_default_config["custom_config"],
        "kwargs": {},
    }

    policy_copy: Policy = policy_cls(
        observation_space=observation_space,
        action_space=action_space,
        model_config=rl_default_config["model_config"],
        custom_config=rl_default_config["custom_config"],
    )

    # create a parameter table
    strategy_spec = StrategySpec(
        identifier="test_parameter_server",
        policy_ids=[f"policy-{i}" for i in range(10)],
        meta_data={
            "policy_cls": policy_cls,
            "kwargs": policy_kwargs,
            "experiment_tag": "test_parameter_server",
        },
    )
    server.create_table(strategy_spec=strategy_spec)

    # try to create repeated table: should jump over
    server.create_table(strategy_spec=strategy_spec)

    # set weights
    server.set_weights(
        spec_id=strategy_spec.id,
        spec_policy_id="policy-1",
        state_dict=policy_copy.state_dict(),
    )

    # retrive weights
    server.get_weights(spec_id=strategy_spec.id, spec_policy_id="policy-1")
