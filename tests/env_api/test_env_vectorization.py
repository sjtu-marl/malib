import importlib
import pytest

from malib.envs import Environment, vector_env
from malib.utils.episode import EpisodeKey


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        ("malib.envs.poker", "PokerParallelEnv", "leduc_poker", {"fixed_player": True}),
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}),
        ("malib.envs.mpe", "MPE", "simple_spread_v2", {"max_cycles": 25}),
        (
            "malib.envs.gr_football",
            "BaseGFootBall",
            "Gfootball",
            {
                "env_name": "academy_run_pass_and_shoot_with_keeper",
                "number_of_left_players_agent_controls": 2,
                "number_of_right_players_agent_controls": 1,
                "representation": "raw",
                "logdir": "",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
                "stacked": False,
            },
        ),
    ],
)
class TestVecEnv:
    @pytest.fixture(autouse=True)
    def _create_cur_instance(self, module_path, cname, env_id, scenario_configs):
        creator = getattr(importlib.import_module(module_path), cname)
        env: Environment = creator(env_id=env_id, scenario_configs=scenario_configs)

        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces

        self.creator = creator
        self.config = {"env_id": env_id, "scenario_configs": scenario_configs}
        self.vec_env = vector_env.VectorEnv(
            observation_spaces,
            action_spaces,
            creator,
            configs={"scenario_configs": scenario_configs, "env_id": env_id},
        )

    def test_add_envs(self):
        self.vec_env.add_envs(num=2)
        assert self.vec_env.num_envs == 2

    def test_env_reset(self):
        self.vec_env.add_envs(num=4)

        rets = self.vec_env.reset(limits=2, fragment_length=100, max_step=25)

        assert len(self.vec_env.active_envs) == 2
        assert self.vec_env._fragment_length == 100
        assert len(rets) == 2, rets
        for env_id, ret in rets.items():
            assert env_id in self.vec_env.active_envs
            for k, agent_v in ret.items():
                for agent in agent_v:
                    if agent == "__all__":
                        continue
                    assert agent in self.vec_env.possible_agents

    def test_env_step(self):
        self.vec_env.add_envs(num=4)

        rets = self.vec_env.reset(limits=2, fragment_length=100, max_step=25)

        act_spaces = self.vec_env.action_spaces

        for _ in range(10):
            actions = {}
            for env_id, ret in rets.items():
                if EpisodeKey.CUR_OBS not in ret:
                    obs_k = EpisodeKey.NEXT_OBS
                else:
                    obs_k = EpisodeKey.CUR_OBS
                live_agents = list(ret[obs_k].keys())
                actions[env_id] = {aid: act_spaces[aid].sample() for aid in live_agents}
            rets = {
                eid: _ret
                for eid, _ret in self.vec_env.step(actions).items()
                if eid in self.vec_env.active_envs
            }


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        # ("malib.envs.poker", "PokerParallelEnv", "leduc_poker", {"fixed_player": True}),
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}),
        ("malib.envs.mpe", "MPE", "simple_spread_v2", {"max_cycles": 25}),
        (
            "malib.envs.gr_football",
            "BaseGFootBall",
            "Gfootball",
            {
                "env_name": "academy_run_pass_and_shoot_with_keeper",
                "number_of_left_players_agent_controls": 2,
                "number_of_right_players_agent_controls": 1,
                "representation": "raw",
                "logdir": "",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
                "stacked": False,
            },
        ),
    ],
)
class TestSubprocVecEnv:
    @pytest.fixture(autouse=True)
    def _create_cur_instance(self, module_path, cname, env_id, scenario_configs):
        creator = getattr(importlib.import_module(module_path), cname)
        env: Environment = creator(env_id=env_id, scenario_configs=scenario_configs)

        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces

        self.creator = creator
        self.config = {"env_id": env_id, "scenario_configs": scenario_configs}
        self.vec_env = vector_env.SubprocVecEnv(
            observation_spaces,
            action_spaces,
            creator,
            configs={"scenario_configs": scenario_configs, "env_id": env_id},
            max_num_envs=4,
        )

    def test_add_envs(self):
        self.vec_env.add_envs(num=2)
        assert self.vec_env.num_envs == 2

        # save destroy
        self.vec_env.close()

    def test_env_step(self):
        self.vec_env.add_envs(num=3)

        rets = self.vec_env.reset(limits=2, fragment_length=100, max_step=25)

        act_spaces = self.vec_env.action_spaces

        # FIXME(ming): may be stuck by the warning: Illegal move made, game terminating with current player losing.
        #   obs['action_mask'] contains a mask of all legal moves that can be chosen
        for _ in range(10):
            actions = {}
            for env_id, ret in rets.items():
                if EpisodeKey.CUR_OBS not in ret:
                    obs_k = EpisodeKey.NEXT_OBS
                else:
                    obs_k = EpisodeKey.CUR_OBS
                live_agents = list(ret[obs_k].keys())
                actions[env_id] = {aid: act_spaces[aid].sample() for aid in live_agents}
            rets = {
                eid: _ret
                for eid, _ret in self.vec_env.step(actions).items()
                if eid in self.vec_env.active_envs
            }

        self.vec_env.close()
