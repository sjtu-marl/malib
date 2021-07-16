import logging
import pytest

from malib.envs.vector_env import VectorEnv
from malib.envs.maatari.env import MAAtari
from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.random.policy import RandomPolicy


@pytest.fixture(scope="session")
def maatari_env_config_set():
    return {
        "basketball_pong": {
            "env_id": "basketball_pong_v2",
            "wrappers": [
                {"name": "resize_v0", "params": [84, 84]},
                {"name": "dtype_v0", "params": ["float32"]},
                {"name": "normalize_obs_v0", "params": {"env_min": 0, "env_max": 1}},
            ],
            "scenario_configs": {"obs_type": "grayscale_image", "num_players": 2},
        }
    }


class TestVecotorMAatari:
    def test_basketball_pong(self, maatari_env_config_set):
        configs = maatari_env_config_set["basketball_pong"]

        env = MAAtari(**configs)
        vector_env = VectorEnv(
            env.observation_spaces, env.action_spaces, MAAtari, configs, num_envs=9
        )
        vector_env.add_envs([env])

        random_policy = {
            aid: RandomPolicy(
                "random", env.observation_spaces[aid], env.action_spaces[aid], {}, {}
            )
            for aid in env.trainable_agents
        }

        assert vector_env.num_envs == 10

        # stepping 10 steps at most
        done, step = False, 0
        rets = vector_env.reset()

        assert Episode.CUR_OBS in rets

        while not done and step < 10:
            actions = {
                k: random_policy[k].compute_actions(obs)[0]
                for k, obs in rets[Episode.CUR_OBS].items()
            }
            assert len(list(actions.values())[0]) == 10
            rets = vector_env.step(actions)

            assert Episode.NEXT_OBS in rets

            rets[Episode.CUR_OBS] = rets[Episode.NEXT_OBS]
            step += 1
            done = all([sum(v) for v in rets[Episode.DONE].values()])
            logging.debug(f"stepping basketball pong at step={step}")
