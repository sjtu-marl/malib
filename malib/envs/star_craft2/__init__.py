from malib.utils.typing import Dict

from .star_craft_env import SC2Env, StatedSC2


def env_desc_gen(env_id: str, scenario_configs: Dict = None):
    env = SC2Env(env_id=env_id, scenari_config=scenario_configs)
    env_desc = {
        "creator": SC2Env,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": {"env_id": env_id, "scenario_configs": scenario_configs or {}},
    }
    env.close()
    return env_desc
