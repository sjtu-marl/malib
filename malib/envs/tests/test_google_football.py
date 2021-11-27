import pytest

from malib.envs.gr_football import BaseGFootBall, ParameterizedSharing
from malib.utils.episode import EpisodeKey


@pytest.mark.parametrize(
    "env_name,n_player_left,n_player_right",
    [("academy_run_pass_and_shoot_with_keeper", 2, 1)],
)
class TestGoogleFootballEnv:
    @pytest.fixture(autouse=True)
    def _init(
        self,
        env_name: str,
        n_player_left: int,
        n_player_right: int,
        use_built_in_GK: bool = True,
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
        }

        self.env = BaseGFootBall(
            env_id="Gfootball",
            use_built_in_GK=use_built_in_GK,
            scenario_configs=scenario_configs,
        )

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

    def test_wrapper(self):
        mapping_func = lambda x: x[:6]
        env = ParameterizedSharing(self.env, mapping_func)

        state_spaces = env.state_spaces
        observation_spaces = env.observation_spaces
        act_spaces = env.action_spaces

        rets = env.reset(max_step=20)
        assert EpisodeKey.CUR_STATE in rets

        for aid, state in rets[EpisodeKey.CUR_STATE].items():
            assert state_spaces[aid].contains(state), (aid, state_spaces, state)

        for _ in range(20):
            actions = {aid: space.sample() for aid, space in act_spaces.items()}
            rets = env.step(actions)

        assert self.env.cnt <= 20
        assert rets[EpisodeKey.DONE]["__all__"], (self.env.cnt, rets[EpisodeKey.DONE])
