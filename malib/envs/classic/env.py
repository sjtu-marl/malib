import importlib

from malib.envs.env import SequentialEnvironment
from malib.backend.datapool.offline_dataset_server import Episode


class Classic(SequentialEnvironment):
    def __init__(self, **configs):
        super(Classic).__init__(**configs)
        self._env = importlib(f"pettingzoo.classic.{configs['env_id']}")
        self._extra_returns = [Episode.ACTION_MASK]

    def last(self):
        observation, reward, done, info = self._env.last()
        action_mask = observation["action_mask"]
        return {
            Episode.CUR_OBS: observation,
            Episode.REWARD: reward,
            Episode.DONE: done,
            Episode.ACTION_MASK: action_mask,
        }
