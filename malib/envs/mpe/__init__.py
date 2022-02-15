from malib.utils.typing import Dict
from .env import MPE


def env_desc_gen(**config):
    env = MPE(**config)
    env_desc = {
        "creator": MPE,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "teams": config.get("teams", {}),
        "config": config,
    }
    env.close()
    return env_desc
