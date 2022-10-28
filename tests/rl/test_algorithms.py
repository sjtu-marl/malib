import os
import pytest
import numpy as np

from malib import rl
from malib.rl.common import policy
from malib.rollout.envs.mdp.env import MDPEnvironment


@pytest.mark.parametrize("algorithm", [rl.pg, rl.a2c, rl.dqn])
@pytest.mark.parametrize(
    "mdp_env_id",
    [
        "one_round_dmdp",
        "two_round_dmdp",
        "one_round_nmdp",
        "two_round_nmdp",
        "multi_round_nmdp",
    ],
)
class TestAlgorithm:
    @pytest.mark.parametrize("evaluation_mode", [False, True])
    def test_policy_construct(self, algorithm, mdp_env_id, evaluation_mode):
        env = MDPEnvironment(env_id=mdp_env_id)
        agent: policy.Policy = algorithm.POLICY(
            env.observation_spaces["agent"],
            env.action_spaces["agent"],
            algorithm.DEFAULT_CONFIG["model_config"],
            algorithm.DEFAULT_CONFIG["custom_config"],
        )

        done = False
        _, obs = env.reset()
        total_rew = 0
        cnt = 0
        while not done:
            obs = {k: agent.preprocessor.transform(v) for k, v in obs.items()}
            actions = {
                k: agent.compute_action(
                    v.reshape(1, -1), act_mask=None, evaluate=evaluation_mode
                )[0][0]
                for k, v in obs.items()
            }
            _, obs, rew, done, info = env.step(actions)
            done = done["__all__"]
            cnt += 1
            total_rew += rew["agent"]
        print(total_rew, cnt)

    def test_trainer_construct(self):
        pass

    def test_optimization(self):
        pass
