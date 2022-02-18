from malib.utils.typing import Dict

from .poker_aec_env import PokerParallelEnv


def env_desc_gen(**config):
    env = PokerParallelEnv(**config)
    env_desc = {
        "creator": PokerParallelEnv,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
    env.close()
    return env_desc
