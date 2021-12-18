from typing import Dict
from gym.spaces import space
from malib.algorithm.maddpg.loss import MADDPGLoss
from malib.algorithm.maddpg.trainer import MADDPGTrainer
from malib.utils.episode import EpisodeKey
from tests.algorithm import AlgorithmTestMixin
from gym import spaces
import numpy as np
from malib.algorithm.maddpg import CONFIG, POLICY

custom_config = CONFIG["policy"]
trainer_config = CONFIG["training"]

custom_config["use_rnn"] = True

model_config = {
    "initialization": {
        "use_orthogonal": True,
        "gain": 1.0,
    },
    "actor": {
        "network": "mlp",
        "layers": [{"units": 8, "activation": "ReLU"}],
        "output": {"activation": False},
    },
    "critic": {
        "network": "mlp",
        "layers": [{"units": 8, "activation": "ReLU"}],
        "output": {"activation": False},
    },
}

test_obs_shape = (3,)
test_action_dim = 2


class TestMADDPG(AlgorithmTestMixin):
    def make_algorithm(self, *args):
        custom_config["global_state_space"] = spaces.Box(
            low=0, high=1, shape=(test_obs_shape[0] + test_action_dim,)
        )
        return POLICY(
            registered_name="MADDPG",
            observation_space=spaces.Box(low=0, high=1, shape=test_obs_shape),
            action_space=spaces.Discrete(n=test_action_dim),
            model_config=model_config,
            custom_config=custom_config,
            env_agent_id="agent_0",
        )

    def make_trainer_and_config(self):
        return MADDPGTrainer("test_trainer"), trainer_config

    def make_loss(self):
        return MADDPGLoss()

    def build_env_inputs(self) -> Dict:
        return {
            EpisodeKey.CUR_OBS: np.zeros((1,) + test_obs_shape),
            EpisodeKey.DONE: np.zeros((1, 1)),
            EpisodeKey.RNN_STATE: self.algorithm.get_initial_state(1),
        }

    def build_train_inputs(self) -> Dict:
        batch_size = 32
        self.trainer.agents = ["agent_0"]
        self.trainer.main_id = "agent_0"
        return {
            "agent_0": {
                EpisodeKey.CUR_OBS: np.zeros((batch_size,) + test_obs_shape),
                EpisodeKey.NEXT_OBS: np.zeros((batch_size,) + test_obs_shape),
                EpisodeKey.DONE: np.zeros((batch_size, 1)),
                EpisodeKey.REWARD: np.zeros((batch_size, 1)),
                EpisodeKey.ACTION: np.zeros((batch_size, 1)),
                EpisodeKey.ACTION_DIST: np.ones((batch_size, test_action_dim))
                / test_action_dim,
                "next_act_by_target": np.zeros((batch_size, test_action_dim)),
            },
        }

    def test_trainer_preprocess(self):
        self.trainer.preprocess(self.build_train_inputs, other_policies={})
