from gym.spaces import space
from malib.algorithm.tests.test_dqn import custom_config
from tests.algorithm import AlgorithmTestMixin
from gym import spaces
import numpy as np
from malib.algorithm.mappo import CONFIG, MAPPO

custom_config = CONFIG['policy']

model_config = {
    'actor': {
        'network': 'mlp',
        'layers': [{'units': 8, 'activation': 'ReLU'}],
        'output': {'activation': False},
    },
    'critic': {
        'network': 'mlp',
        'layers': [{'units': 8, 'activation': 'ReLU'}],
        'output': {'activation': False},
    }
}

test_obs_shape = (3,)
test_action_dim = 2

class TestDQN(AlgorithmTestMixin):
    def make_algorithm(self, *args):
        return MAPPO(
            registered_name='MAPPO',
            observation_space=spaces.Box(low=0, high=1, shape=test_obs_shape),
            action_space=spaces.Discrete(n=test_action_dim),
            model_config=model_config,
            custom_config=custom_config
        )

    def build_env_inputs(self) -> Dict:
        return {
            'observation': np.zeros((1,)+ test_obs_shape),
            'rnn_state': []
        }