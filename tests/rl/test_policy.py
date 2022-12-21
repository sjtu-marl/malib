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

from typing import Dict, Any
from functools import partial

import pytest
import gym

from gym import spaces

from malib.rl.common.policy import Policy


class FakePolicy(Policy):
    def compute_action(
        self, observation, act_mask, evaluate: bool, hidden_state: Any = None, **kwargs
    ):
        super().compute_action(observation, act_mask, evaluate, hidden_state, **kwargs)
        return "checked"

    def coordinate(self, state, message: Any) -> Any:
        super().coordinate(state, message)
        return "checked"


@pytest.mark.parametrize(
    "observation_space",
    [spaces.Box(low=-1.0, high=1.0, shape=(4,)), spaces.Discrete(3)],
)
@pytest.mark.parametrize(
    "action_space",
    [
        spaces.Box(low=-1.0, high=1.0, shape=(4,)),
        spaces.Discrete(3),
        spaces.MultiBinary(10),
    ],
)
@pytest.mark.parametrize("model_config", [{}])
@pytest.mark.parametrize(
    "custom_config",
    [
        {"preprocess_mode": "flatten", "use_cuda": True},
        {"preprocess_mode": "flatten", "use_cuda": False},
    ],
)
def test_interface_calling(
    observation_space: gym.Space,
    action_space: gym.Space,
    model_config: Dict[str, Any],
    custom_config: Dict[str, Any],
):

    policy_caller = partial(
        FakePolicy, observation_space, action_space, model_config, custom_config
    )
    if isinstance(action_space, spaces.MultiBinary):
        with pytest.raises(NotImplementedError):
            policy_caller()
    else:
        policy = policy_caller()
        model_config = policy.model_config
        assert model_config == model_config

        if custom_config.get("use_cuda", False):
            assert "cuda" in policy.device.type
        else:
            assert "cpu" in policy.device.type

        # resetting
        policy.register_state("dd", "_target_actor")
        policy.target_actor = "abc"
        assert policy.target_actor == "abc"

        policy.register_state("dds", "_target_critic")
        policy.target_critic = "cbd"
        assert policy.target_critic == "cbd"

        policy.register_state([1, 2, 3], name="test_x")
        assert "test_x" in policy._state_handler_dict
        policy.deregister_state("test_x")
        assert "test_x" not in policy._state_handler_dict

        # save tmp
        policy.save("./pp.pkl")
        policy.load("./pp.pkl")
        policy.reset()
        FakePolicy.copy(policy, replacement={})

        for device in ["cpu", "cuda"]:
            for use_copy in [True, False]:
                policy.to(device=device, use_copy=use_copy)
