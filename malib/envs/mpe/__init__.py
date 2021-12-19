from malib.utils.typing import Dict
from .env import MPE


def env_desc_gen(env_id: str, scenario_configs: Dict = None):
    if "teams" in scenario_configs:
        teams = scenario_configs.pop("teams")
    else:
        teams = None
    env = MPE(env_id=env_id, scenario_config=scenario_configs)
    env_desc = {
        "creator": MPE,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "teams": teams,
        "config": {
            "env_id": env_id,
            "scenario_configs": scenario_configs or {},
        },
    }
    env.close()
    return env_desc
