from malib.utils.typing import Dict

from .star_craft_env import StatedSC2


def env_desc_gen(**config):
    env = StatedSC2(**config)
    env_desc = {
        "creator": StatedSC2,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "global_state_spaces": env.state_spaces,
        "config": config,
        "teams": env.group_to_agents,
    }
    env.close()
    return env_desc
