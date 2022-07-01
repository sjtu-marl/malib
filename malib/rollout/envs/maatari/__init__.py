from .env import MAAtari


def env_desc_gen(**config):
    env = MAAtari(**config)
    env_desc = {
        "creator": MAAtari,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
    env.close()
    return env_desc
