from typing import Dict
from gym.spaces import space
from malib.algorithm.mappo.loss import MAPPOLoss
from malib.algorithm.mappo.trainer import MAPPOTrainer
from malib.utils.episode import EpisodeKey
from tests.algorithm import AlgorithmTestMixin
from gym import spaces
import numpy as np
from malib.algorithm.mappo import CONFIG, MAPPO

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


class TestMAPPO(AlgorithmTestMixin):
    def make_algorithm(self, *args):
        custom_config["global_state_space"] = {
            "agent_0": spaces.Box(low=0, high=1, shape=test_obs_shape)
        }
        return MAPPO(
            registered_name="MAPPO",
            observation_space=spaces.Box(low=0, high=1, shape=test_obs_shape),
            action_space=spaces.Discrete(n=test_action_dim),
            model_config=model_config,
            custom_config=custom_config,
            env_agent_id="agent_0",
        )

    def make_trainer_and_config(self):
        return MAPPOTrainer("test_trainer"), trainer_config

    def make_loss(self):
        return MAPPOLoss()

    def build_env_inputs(self) -> Dict:
        return {
            "observation": np.zeros((4,) + test_obs_shape),
            "rnn_state": self.algorithm.get_initial_state(batch_size=4),
            "done": np.zeros((4, 1)),
        }

    def build_train_inputs(self) -> Dict:
        n_agent, batch_size, traj_len = 4, 32, 100
        num_rnn_layer = custom_config["rnn_layer_num"]
        rnn_states = self.algorithm.get_initial_state(
            batch_size=n_agent * batch_size * traj_len
        )
        actor_rnn_state = rnn_states[0].reshape(
            (batch_size, traj_len, n_agent, num_rnn_layer, -1)
        )
        critic_rnn_state = rnn_states[1].reshape(
            (batch_size, traj_len, n_agent, num_rnn_layer, -1)
        )

        return {
            EpisodeKey.CUR_OBS: np.zeros(
                (
                    batch_size,
                    traj_len,
                    n_agent,
                )
                + test_obs_shape
            ),
            EpisodeKey.CUR_STATE: np.zeros(
                (
                    batch_size,
                    traj_len,
                    n_agent,
                )
                + test_obs_shape
            ),
            EpisodeKey.DONE: np.zeros((batch_size, traj_len, n_agent, 1)),
            EpisodeKey.REWARD: np.zeros((batch_size, traj_len, n_agent, 1)),
            EpisodeKey.ACTION: np.zeros((batch_size, traj_len, n_agent, 1)),
            EpisodeKey.ACTION_DIST: np.ones(
                (batch_size, traj_len, n_agent, test_action_dim)
            )
            / test_action_dim,
            EpisodeKey.STATE_VALUE: np.zeros((batch_size, traj_len, n_agent, 1)),
            "return": np.zeros((batch_size, traj_len, n_agent, 1)),
            EpisodeKey.ACTION_MASK: np.zeros((batch_size, traj_len, n_agent, 1)),
            EpisodeKey.RNN_STATE + "_0": actor_rnn_state,
            EpisodeKey.RNN_STATE + "_1": critic_rnn_state,
        }
