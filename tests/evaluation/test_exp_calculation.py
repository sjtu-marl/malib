import pytest
import pyspiel

from gym import spaces
from pytest_mock import MockerFixture

from malib.utils.typing import BehaviorMode
from malib.gt.algos import exploitability
from malib.algorithm.random import RandomPolicy


@pytest.mark.parametrize("game_name", ["leduc_poker"])
def test_exp_calculation(game_name: str, mocker: MockerFixture):
    game = pyspiel.load_game(game_name)

    policy = RandomPolicy(
        "test",
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        action_space=spaces.Discrete(3),
        model_config={},
        custom_config={},
    )

    # mock malib policy API
    def transform_func(obs):
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs
        # only action_mask is required
        return obs["action_mask"]

    def compute_action(observation: list, **kwargs):
        assert len(observation) == 1
        assert "behavior_mode" in kwargs
        assert "legal_actions_list" in kwargs
        assert kwargs["behavior_mode"] == BehaviorMode.EXPLOITATION, kwargs[
            "behavior_mode"
        ]

        legal_actions_list = kwargs["legal_actions_list"]
        probs = [{idx: 1.0 / len(legal_actions_list) for idx in legal_actions_list}]
        return None, probs, None

    policy.preprocessor.transform = mocker.patch.object(
        policy.preprocessor, "transform", side_effect=transform_func
    )
    policy.compute_action = mocker.patch.object(
        policy, "compute_action", side_effect=compute_action
    )
    instance = exploitability.build_open_spiel_policy(policy, game)
    assert isinstance(instance, exploitability.PolicyFromCallable)

    # build fake population
    populations = {
        i: {f"policy_{j}": policy for j in range(2)} for i in range(game.num_players())
    }
    mixture = {
        i: {f"policy_{j}": 0.5 for j in range(2)} for i in range(game.num_players())
    }
    nash = exploitability.measure_exploitability(
        game_name=game_name, populations=populations, policy_mixture_dict=mixture
    )
