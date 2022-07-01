from .env import DummyEnv


def env_desc_gen(**config):
    env = DummyEnv(**config)
    return {
        "creator": DummyEnv,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
