from malib.utils.typing import Dict

from .star_craft_env import SC2Env


def env_desc_gen(**config):
    env = SC2Env(**config)
    env_desc = {
        "creator": SC2Env,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
    env.close()
    return env_desc
