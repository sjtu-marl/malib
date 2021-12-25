import numpy as np
import pytest

from malib.envs.gr_football import BaseGFootBall
from malib.envs.gr_football.env import ParameterSharing
from malib.envs.gr_football.wrappers import GroupedGFBall
from malib.utils.episode import EpisodeKey


@pytest.mark.parametrize(
    "env_name,n_player_left,n_player_right,use_built_in_GK, action_set",
    [
        ("5_vs_5", 4, 0, True, "default"),
        ("5_vs_5", 5, 5, True, "v2"),  # 5v5 failed
        ("5_vs_5", 5, 0, True, "v2"),
        ("5_vs_5", 4, 0, False, "default"),
        ("5_vs_5", 5, 5, False, "v2"),  # 5v5 failed
        ("5_vs_5", 5, 0, False, "v2"),
    ],
    scope="class",
)
class TestGoogleFootballEnv:
    @pytest.fixture(autouse=True)
    def _init(
        self,
        env_name: str,
        n_player_left: int,
        n_player_right: int,
        action_set: str,
        use_built_in_GK: bool,
    ):
        scenario_configs = {
            "env_name": env_name,
            "number_of_left_players_agent_controls": n_player_left,
            "number_of_right_players_agent_controls": n_player_right,
            "representation": "raw",
            "logdir": "",
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "stacked": False,
            "other_config_options": {"action_set": action_set},
        }

        self.env = BaseGFootBall(
            env_id="Gfootball",
            use_built_in_GK=use_built_in_GK,
            scenario_configs=scenario_configs,
        )

        self.env.seed()

        self.env_id = "Gfootball"
        self.use_built_in_GK = use_built_in_GK
        self.scenario_configs = scenario_configs

    def test_env_api(self):
        rets = self.env.reset(max_step=20)
        act_spaces = self.env.action_spaces
        assert EpisodeKey.CUR_OBS in rets

        for _ in range(20):
            action = {aid: space.sample() for aid, space in act_spaces.items()}
            rets = self.env.step(action)

        assert self.env.cnt <= 20
        assert rets[EpisodeKey.DONE]["__all__"], (self.env.cnt, rets[EpisodeKey.DONE])

        print(self.env.collect_info())

    def test_parameter_sharing_wrapper(self):
        mapping_func = lambda x: x[:6]
        env = ParameterSharing(self.env, mapping_func)

        state_spaces = env.state_spaces
        observation_spaces = env.observation_spaces
        act_spaces = env.action_spaces

        rets = env.reset(max_step=20)
        assert EpisodeKey.CUR_STATE in rets

        for aid, obs in rets[EpisodeKey.CUR_OBS].items():
            assert obs.shape[1] == observation_spaces[aid].shape[0]

        for aid, state in rets[EpisodeKey.CUR_STATE].items():
            assert len(state.shape) == 2
            assert state_spaces[aid].shape[0] == state.shape[1], (
                aid,
                state_spaces,
                state,
            )

        for _ in range(20):
            actions = {
                aid: np.asarray(
                    [space.sample()] * rets[EpisodeKey.CUR_OBS][aid].shape[0], dtype=int
                )
                for aid, space in act_spaces.items()
            }
            rets = env.step(actions)
            # update next to cur
            rets[EpisodeKey.CUR_OBS] = rets[EpisodeKey.NEXT_OBS]
            rets[EpisodeKey.CUR_STATE] = rets[EpisodeKey.NEXT_STATE]

        assert self.env.cnt <= 20
        assert rets[EpisodeKey.DONE]["__all__"], (self.env.cnt, rets[EpisodeKey.DONE])
        env.close()

    @pytest.mark.parametrize(
        "group_func",
        [
            lambda agent_id: agent_id,
        ],
    )
    def test_group_wrapper(self, group_func):
        env = GroupedGFBall(self.env, group_func)

        state_spaces = env.state_spaces
        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces

        # check whether the group rule matches the group_func
        possible_agents = self.env.possible_agents
        for aid in possible_agents:
            group_pred = env.group_rule(aid)
            group_target = group_func(aid)
            assert group_pred == group_target, (group_pred, group_target)
            assert aid in state_spaces
            assert aid in observation_spaces
            assert aid in action_spaces

        rets = env.reset(max_step=20)
        assert EpisodeKey.CUR_STATE in rets

        for aid, obs in rets[EpisodeKey.CUR_OBS].items():
            assert observation_spaces[aid].contains(obs)

        for aid, state in rets[EpisodeKey.CUR_STATE].items():
            assert state_spaces[aid].contains(state)

        for _ in range(20):
            actions = {aid: space.sample() for aid, space in action_spaces.items()}
            rets = env.step(actions)
            # update next to cur
            rets[EpisodeKey.CUR_OBS] = rets[EpisodeKey.NEXT_OBS]
            rets[EpisodeKey.CUR_STATE] = rets[EpisodeKey.NEXT_STATE]

        assert self.env.cnt <= 20
        assert rets[EpisodeKey.DONE]["__all__"], (self.env.cnt, rets[EpisodeKey.DONE])

        env.close()
