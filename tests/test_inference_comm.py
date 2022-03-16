import pytest
import ray
import pprint

from malib.settings import PARAMETER_SERVER_ACTOR
from malib.utils.typing import BehaviorMode, Dict, AgentID, Any

from malib.rollout.inference_server import InferenceWorkerSet
from malib.rollout.rollout_func import Stepping
from malib.algorithm.random.policy import RandomPolicy
from malib.envs import gr_football
from tests.parameter_server import FakeParameterServer


@pytest.fixture(scope="session")
def env_desc():
    return gr_football.env_desc_gen(
        {
            "env_id": "PSGFootball",
            "use_built_in_GK": True,
            "scenario_configs": {
                "env_name": "5_vs_5",
                "number_of_left_players_agent_controls": 4,
                "number_of_right_players_agent_controls": 0,
                "representation": "raw",
                "stacked": False,
                "logdir": "/tmp/football/malib_psro",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
            },
        }
    )


@pytest.fixture(scope="session")
def inference_actors(env_desc):
    if not ray.is_initialized():
        ray.init()

    parameter_server = FakeParameterServer.options(name=PARAMETER_SERVER_ACTOR).remote()

    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]

    return {
        aid: InferenceWorkerSet.remote(
            observation_space=obs_space,
            action_space=act_spaces[aid],
            parameter_server=parameter_server,
        )
        for aid, obs_space in obs_spaces.items()
    }


@pytest.mark.parametrize(
    "runtime_config",
    [
        {
            "behavior_mode": BehaviorMode.EXPLOITATION,
            "num_episodes": 2,
            "max_step": 3000,
            "fragment_length": 6000,
            "behavior_policies": {"team0": []},
        }
    ],
)
@pytest.mark.parametrize("flag", ["rollout", "evaluation"])
def test_comm(
    inference_actors: Dict[AgentID, InferenceWorkerSet],
    env_desc: Dict[str, Any],
    runtime_config: Dict[str, Any],
    flag: str,
):

    client = Stepping.as_remote().remote(env_desc, use_subproc_env=False)

    for actor in inference_actors.values():
        actor.reset_comm(client)

    res = ray.get(
        client.run.remote(
            inference_actors,
            desc={"runtime_config": runtime_config, "flag": flag},
        )
    )

    pprint.pprint(res)
