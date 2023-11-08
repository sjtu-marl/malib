import pytest
import argparse

from malib.rl.random import RandomPolicy
from malib.common.strategy_spec import StrategySpec
from malib.rollout.envs.mdp import env_desc_gen


class TestStrategySpec:
    @pytest.fixture
    def spec(self) -> StrategySpec:
        env_desc = argparse.Namespace(**env_desc_gen(env_id="one_round_dmdp"))
        return StrategySpec(
            policy_cls=RandomPolicy,
            observation_space=env_desc.observation_spaces["default"],
            action_space=env_desc.action_spaces["default"],
            identifier="random",
        )

    def test_existing_policy_register(self, spec: StrategySpec):
        spec.register_policy_id("random0")
        assert len(spec) == 1
        with pytest.raises(KeyError):
            spec.register_policy_id("random0")
        spec.register_policy_id("random2")
        assert len(spec) == 2
        assert spec.policy_ids == ("random0", "random2")

    def test_policy_generation(self, spec: StrategySpec):
        model = spec.gen_policy(device="cpu")
        model = spec.gen_policy(device="cuda:0")

    def test_policy_dist_reset(self, spec: StrategySpec):
        spec.register_policy_id("random0")
        spec.register_policy_id("random2")
        spec.update_prob_list({"random0": 0.5, "random2": 0.5})

        with pytest.raises(ValueError) as excinfo:
            spec.update_prob_list({"random0": 0.5, "random2": 0.5, "random3": 0.5})

        with pytest.raises(ValueError) as excinfo:
            spec.update_prob_list({"random0": 0.5, "random2": 0.1})

    def test_policy_sampling(self, spec: StrategySpec):
        spec.register_policy_id("random0")
        spec.register_policy_id("random2")
        spec.sample()
        spec.update_prob_list({"random0": 1, "random2": 0})
        for _ in range(10):
            pid = spec.sample()
            assert "random0" == pid, pid
