from rollout_function import grf_simultaneous

from malib.envs.gr_football import default_config
from malib.envs.gr_football.env import BaseGFootBall, ParameterSharingWrapper
from malib.algorithm.mappo.policy import MAPPO
import yaml
from malib.envs.subproc_vec_env import SubprocVecEnv


if __name__ == "__main__":
    env_fn = lambda: ParameterSharingWrapper(
        BaseGFootBall(**default_config), lambda x: x[:6]
    )
    env = env_fn()

    cfg = yaml.load(open("examples/mappo_gfootball/mappo_5_vs_5.yaml"))

    custom_cfg = cfg["algorithms"]["MAPPO"]["custom_config"]
    custom_cfg.update({"global_state_space": env.state_space})

    policies = {
        aid: MAPPO(
            "MAPPO",
            env.observation_spaces[aid],
            env.action_spaces[aid],
            cfg["algorithms"]["MAPPO"]["model_config"],
            custom_cfg,
            env_agent_id=aid,
        )
        for aid in env.possible_agents
    }

    vec_env = SubprocVecEnv(
        env.observation_spaces,
        env.action_spaces,
        env_fn,
        {},
        num_envs=3,
        fragment_length=3001,
    )

    grf_simultaneous(
        vec_env,
        3,
        3001,
        None,
        policies,
        None,
    )
