from typing import Dict
from malib.algorithm.random import RandomPolicy, RandomTrainer
from malib.algorithm.common.loss_func import LossFunc
from malib.utils.episode import EpisodeKey
from tests.algorithm import AlgorithmTestMixin
from gym import spaces
import numpy as np

trainer_config = {"optimizer": "Adam"}
custom_config = {}

model_config = {
    "actor": {
        "network": "mlp",
        "layers": [{"units": 64, "activation": "ReLU"}],
        "output": {"activation": False},
    },
    "critic": {
        "network": "mlp",
        "layers": [{"units": 64, "activation": "ReLU"}],
        "output": {"activation": False},
    },
}


test_obs_shape = (3,)
test_action_dim = 2


class TestDDPG(AlgorithmTestMixin):
    def make_algorithm(self, *args):
        return RandomPolicy(
            registered_name="RANDOM",
            observation_space=spaces.Box(low=0, high=1, shape=test_obs_shape),
            action_space=spaces.Discrete(n=test_action_dim),
            model_config=model_config,
            custom_config=custom_config,
        )

    def make_trainer_and_config(self):
        return (
            RandomTrainer("Random"),
            trainer_config,
        )

    def make_loss(self):
        return None

    def build_env_inputs(self) -> Dict:
        return {
            EpisodeKey.CUR_OBS: np.zeros((1,) + test_obs_shape),
            EpisodeKey.RNN_STATE: self.algorithm.get_initial_state(1),
        }

    def build_train_inputs(self) -> Dict:
        batch_size = 32
        return {
            EpisodeKey.CUR_OBS: np.zeros((batch_size,) + test_obs_shape),
            EpisodeKey.NEXT_OBS: np.zeros((batch_size,) + test_obs_shape),
            EpisodeKey.ACTION: np.zeros((batch_size, 1)),
            EpisodeKey.ACTION_DIST: np.ones((batch_size, test_action_dim))
            / test_action_dim,
            EpisodeKey.DONE: np.zeros((batch_size, 1)),
            EpisodeKey.REWARD: np.zeros((batch_size, 1)),
        }

    def test_loss_reset(self):
        pass

    def test_trainer_optimize(self):
        pass

    def test_trainer_reset(self):
        pass