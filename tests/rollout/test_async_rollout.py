import importlib
import pytest

from malib.envs import Environment
from tests.env_api.utils import build_dummy_agent_interfaces
from malib.rollout.rollout_func import (
    Stepping,
    _process_policy_outputs,
    _process_environment_returns,
    _do_policy_eval,
)


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs,use_subproc_env",
    [
        (
            "malib.envs.poker",
            "PokerParallelEnv",
            "leduc_poker",
            {"fixed_player": True},
            False,
        ),
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}, False),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}, False),
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
            False,
        ),
    ],
)
class TestRollout:
    @pytest.fixture(autouse=True)
    def _init(
        self, module_path, cname, env_id, scenario_configs, use_subproc_env: bool
    ):
        creator = getattr(importlib.import_module(module_path), cname)
        env: Environment = creator(env_id=env_id, scenario_configs=scenario_configs)

        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces

        agent_interfaces = build_dummy_agent_interfaces(
            observation_spaces, action_spaces
        )

        env_desc = {
            "creator": creator,
            "config": {"env_id": env_id, "scenario_configs": scenario_configs},
        }

        self.env_desc = env_desc
        self.agent_interfaces = agent_interfaces
        self.use_subproc_env = use_subproc_env

    def test_process_env_return(self):
        pass
        # _process_environment_returns()

    def test_stepping(self):
        stepping = Stepping({}, self.env_desc, use_subproc_env=self.use_subproc_env)

        task_type, rollout_results = stepping.run(
            self.agent_interfaces,
            fragment_length=100,
            desc={
                "flag": "rollout",
                "policy_distribution": None,
                "behavior_policies": None,
                "num_episodes": 2,
                "max_step": 25,
            },
            callback=None,
            buffer_desc=None,
        )
        print("task_type: {}\neval_info: {}".format(task_type, rollout_results))
