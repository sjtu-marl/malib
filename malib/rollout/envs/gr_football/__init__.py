from .env import BaseGFootBall, ParameterSharing


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


def creator(**kwargs):
    base = BaseGFootBall(**kwargs)
    if kwargs.get("enable_sharing", False):
        return ParameterSharing(base, default_sharing_mapping)
    else:
        return base


def env_desc_gen(config):
    env = creator(**config)
    env_desc = {
        "creator": creator,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": config,
    }
    if hasattr(env, "state_spaces"):
        env_desc["state_spaces"] = env.state_spaces
    env.close()
    return env_desc
