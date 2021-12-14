"""Basic class for algorithm tests"""

from typing import Callable, Dict

import torch
from malib.algorithm.common import policy, model, loss_func, trainer

import pytest


class AlgorithmTestMixin:
    """The Mixin class for algorithm tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up configs for build algorithms, environments to be tested on."""
        self._algorithm_to_test = self.make_algorithm()

    def make_algorithm(self, *args):
        """Build the instance of algorithm to be tested."""
        raise NotImplementedError

    def build_env_inputs(self) -> Dict:
        """Build dummy inputs for compute_action"""
        raise NotImplementedError

    @property
    def algorithm(self) -> policy.Policy:
        return self._algorithm_to_test

    def assertRNNStates(self, states):
        if states:
            assert len(states) == 2
            # self.assertEqual(states[0].shape[:-1], states[1].shape[:-1])

    # Test Cases
    # ---------------------------------------------------------------------------

    def test_compute_action(self):
        action, action_probs, rnn_states = self.algorithm.compute_action(
            **self.build_env_inputs()
        )
        assert len(action.shape) + 1 == len(action_probs.shape)
        self.assertRNNStates(rnn_states)

    def test_initial_state(self):
        states = self.algorithm.get_initial_state(batch_size=1)
        self.assertRNNStates(states)

    def test_actor_critic(self):
        self.algorithm.actor
        self.algorithm.critic

    def test_load_state_dict(self):
        state_dict = self.algorithm.state_dict()
        self.algorithm.load_state(state_dict)

    def test_reset(self):
        self.algorithm.reset()

    def test_to_device(self):
        device = self.algorithm.device
        assert isinstance(device, torch.device)
        self.algorithm.to_device(device)

    def test_exploration_callback(self):
        assert (
            isinstance(self.algorithm.exploration_callback, Callable)
            or self.algorithm.exploration_callback is None
        )

    def test_description(self):
        desc = self.algorithm.description
        for k in [
            "registered_name",
            "observation_space",
            "action_space",
            "model_config",
            "custom_config",
        ]:
            assert k in desc
