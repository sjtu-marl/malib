import gym
import numpy as np

from malib.utils.typing import Dict, AgentID
from malib.utils.episode import Episode

from .env import GymEnv


def env_desc_gen(env_id: str, scenario_configs: Dict = None):
    env = GymEnv(env_id=env_id, scenario_config=scenario_configs)
    env_desc = {
        "creator": GymEnv,
        "possible_agents": env.possible_agents,
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
        "config": {
            "env_id": env_id,
            "scenario_configs": scenario_configs or {},
        },
    }
    env.close()
    return env_desc


def basic_sampler_config(
    observation_spaces: Dict[AgentID, gym.Space],
    action_spaces: Dict[AgentID, gym.Space],
    preprocessor: object,
    capacity: int = 1000,
    learning_starts: int = 64,
):
    # for homogeneous agent
    action_space = list(action_spaces.values())[0]
    observation_space = list(observation_spaces.values())[0]
    sampler_config = {
        "dtypes": {
            Episode.REWARD: np.float,
            Episode.NEXT_OBS: np.float,
            Episode.DONE: np.bool,
            Episode.CUR_OBS: np.float,
            Episode.ACTION: np.int
            if isinstance(action_space, gym.spaces.Discrete)
            else np.float,
            Episode.ACTION_DIST: np.float,
        },
        "data_shapes": {
            Episode.REWARD: (),
            Episode.NEXT_OBS: preprocessor.shape,
            Episode.DONE: (),
            Episode.CUR_OBS: preprocessor.shape,
            Episode.ACTION: ()
            if isinstance(action_space, gym.spaces.Discrete)
            else action_space.shape,
            Episode.ACTION_DIST: (action_space.n,)
            if isinstance(action_space, gym.spaces.Discrete)
            else action_space.shape,
        },
        "capacity": capacity,
        "learning_starts": learning_starts,
    }
    return sampler_config
