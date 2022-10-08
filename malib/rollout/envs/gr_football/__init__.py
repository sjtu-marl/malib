# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

from types import LambdaType
from typing import Dict, List

from malib.utils.typing import AgentID
from .env import GRFootball
from .wrappers import GroupedFootball


default_sharing_mapping = lambda x: x[:6]
DEFAULT_ENV_CONNFIG = {
    # env building config
    "env_id": "Gfootball",
    "use_built_in_GK": True,
    "scenario_configs": {
        "env_name": "5_vs_5",
        "number_of_left_players_agent_controls": 4,
        "number_of_right_players_agent_controls": 4,
        "representation": "raw",
        "logdir": "",
        "write_goal_dumps": False,
        "write_full_episode_dumps": False,
        "render": False,
        "stacked": False,
    },
}


def env_desc_gen(config, group: bool = False, agent_group_mapping: LambdaType = None):
    if config is None:
        config = DEFAULT_ENV_CONNFIG

    if not group:
        env = GRFootball(**config)
        env_desc = {"creator": GRFootball}
    else:
        env = GroupedFootball(
            agent_group_mapping=default_sharing_mapping or agent_group_mapping, **config
        )
        env_desc = {"creator": GroupedFootball}
        config["agent_group_mapping"] = default_sharing_mapping or agent_group_mapping

    env_desc.update(
        {
            "possible_agents": env.possible_agents,
            "action_spaces": env.action_spaces,
            "observation_spaces": env.observation_spaces,
            "config": config,
        }
    )

    if hasattr(env, "state_spaces"):
        env_desc["state_spaces"] = env.state_spaces

    env.close()

    return env_desc
