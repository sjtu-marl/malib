from malib.utils.typing import Dict

from .env import GymEnv


def env_desc_gen(**config):
    env = GymEnv(**config)
    env_desc = {
        "creator": GymEnv,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
    env.close()
    return env_desc
