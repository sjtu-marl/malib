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
