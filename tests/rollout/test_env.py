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

from pytest_mock import MockerFixture
from malib.rollout import envs


@pytest.mark.parametrize(
    "env_module,env_id,scenario_configs",
    [
        [envs.pettingzoo, "atari.basketball_pong_v3", {}],
        [envs.pettingzoo, "atari.basketball_pong_v3", {"parallel_simulate": False}],
        [envs.gym, "CartPole-v1", {}],
        [envs.mdp, "one_round_dmdp", {}],
        [envs.open_spiel, "kuhn_poker", {}],
    ],
)
def test_env_api(mocker: MockerFixture, env_module, env_id, scenario_configs):
    assert hasattr(env_module, "env_desc_gen")
    env_desc_gen = env_module.env_desc_gen
    desc = env_desc_gen(env_id=env_id, scenario_configs=scenario_configs)

    env = desc["creator"](**desc["config"])
    action_spaces = env.action_spaces

    _, observations = env.reset(max_step=10)
    done = False

    cnt = 0
    while not done:
        actions = {k: action_spaces[k].sample() for k in observations.keys()}
        _, observations, rewards, dones, infos = env.step(actions)
        done = dones["__all__"]
        cnt += 1
