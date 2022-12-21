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

from typing import List, Dict, Any

import pytest
import numpy as np

from gym import spaces

from malib.utils.typing import AgentID
from malib.rollout.envs.gym import env_desc_gen as gym_env_gen
from malib.rollout.envs.mdp import env_desc_gen as mdp_env_gen
from malib.rollout.envs.open_spiel import env_desc_gen as open_spien_env_gen
from malib.rollout.envs.vector_env import VectorEnv, _RemoteEnv


def construct_vector_env(env_desc: Dict[str, Any], preset_num_envs: int):
    return VectorEnv(
        observation_spaces=env_desc["observation_spaces"],
        action_spaces=env_desc["action_spaces"],
        creator=env_desc["creator"],
        configs=env_desc["config"],
        preset_num_envs=preset_num_envs,
    )


@pytest.mark.parametrize("preset_num_envs", [1, 3])
@pytest.mark.parametrize(
    "env_desc",
    [
        gym_env_gen(env_id="CartPole-v1"),
        mdp_env_gen(env_id="two_round_dmdp"),
        open_spien_env_gen(env_id="leduc_poker"),
    ],
)
class TestVectorEnv:
    def test_env_add(self, env_desc: Dict[str, Any], preset_num_envs: int):
        venv = construct_vector_env(env_desc, preset_num_envs)

        # add existing env instances
        envs = [env_desc["creator"](**env_desc["config"]) for _ in range(2)]
        venv.add_envs(envs)

        assert venv.num_envs == preset_num_envs + len(envs), (
            venv.num_envs,
            preset_num_envs + len(envs),
        )

        # create new instances
        venv.add_envs(num=1)

        assert venv.num_envs == preset_num_envs + len(envs) + 1, (
            venv.num_envs,
            preset_num_envs + len(envs) + 1,
        )

        venv.close()

    def test_from_envs(self, env_desc: Dict[str, Any], preset_num_envs: int):
        envs = [
            env_desc["creator"](**env_desc["config"]) for _ in range(preset_num_envs)
        ]
        vector_env = VectorEnv.from_envs(envs, config=env_desc["config"])
        assert vector_env.num_envs == preset_num_envs, (
            vector_env.num_envs,
            preset_num_envs,
        )

    @pytest.mark.parametrize("max_step", [20, 100])
    def test_env_step(
        self, max_step: int, env_desc: Dict[str, Any], preset_num_envs: int
    ):
        venv = construct_vector_env(env_desc, preset_num_envs)
        fragment_length = max_step * preset_num_envs

        rets = venv.reset(fragment_length=fragment_length, max_step=max_step)
        action_spaces: Dict[str, spaces.Space] = venv.envs[0].action_spaces
        observation_spaces: Dict[str, spaces.Space] = venv.envs[0].observation_spaces
        agents = venv.envs[0].possible_agents

        assert set(agents) == set(venv.possible_agents), (agents, venv.possible_agents)

        while not venv.is_terminated():
            # TODO(ming): should consider action mask
            observation_list = [ret[1] for ret in rets]
            actions = {agent: [] for agent in venv.possible_agents}
            for observations in observation_list:
                for k, v in observations.items():
                    if isinstance(v, dict) and "action_mask" in v:
                        action_mask = v["action_mask"]
                        legal_actions = np.where(np.asarray(action_mask) > 0)[0]
                        if len(legal_actions) > 0:
                            action = np.random.choice(legal_actions)
                        else:
                            action = action_spaces[k].sample()
                    else:
                        action = action_spaces[k].sample()
                    actions[k].append(action)
            actions = {k: np.stack(v) for k, v in actions.items()}
            rets = venv.step(actions=actions)

        assert venv.step_cnt >= fragment_length, (venv.step_cnt, fragment_length)
        cached_episode_infos = venv.collect_info()
        assert len(cached_episode_infos) >= 1, len(cached_episode_infos)
