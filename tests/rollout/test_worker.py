import time
import pytest
import gym


from pytest_mock import MockerFixture
from pytest import MonkeyPatch

from malib.utils.typing import (
    Dict,
    Any,
    TaskDescription,
    TaskType,
)

from tests import ServerMixin, FakeAgentInterface, FakeStepping


def simple_env_desc():
    n_agents = 2
    agent_ids = [f"agent_{i}" for i in range(n_agents)]
    action_spaces = dict(
        zip(agent_ids, [gym.spaces.Discrete(2) for i in range(n_agents)])
    )
    obs_spaces = dict(
        zip(
            agent_ids,
            [gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)) for _ in range(n_agents)],
        )
    )

    return {
        "creator": None,
        "possible_agents": agent_ids,
        "action_spaces": action_spaces,
        "observation_spaces": obs_spaces,
        "config": {"env_id": "test", "scenario_configs": None},
    }


@pytest.mark.parametrize(
    "env_desc,kwargs",
    [
        (
            simple_env_desc(),
            {
                "num_rollout_actors": 2,
                "num_eval_actors": 1,
                "exp_cfg": {},
                "use_subproc_env": False,
                "batch_mode": False,
                "postprocessor_types": ["default"],
            },
        )
    ],
    scope="class",
)
class TestRolloutWorker(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(
        self,
        env_desc: Dict[str, Any],
        kwargs: Dict,
        mocker: MockerFixture,
        monkeypatch: MonkeyPatch,
    ):
        self.locals = locals()
        self.coordinator = self.init_coordinator()
        self.parameter_server = self.init_parameter_server()
        self.dataset_server = self.init_dataserver()
        # mute remote logger
        monkeypatch.setattr("malib.settings.USE_REMOTE_LOGGER", False)
        monkeypatch.setattr("malib.settings.USE_MONGO_LOGGER", False)
        # XXX(ming): mock AgentInterface directly will raise deep recursive error here.
        monkeypatch.setattr(
            "malib.rollout.base_worker.AgentInterface", FakeAgentInterface
        )
        monkeypatch.setattr(
            "malib.rollout.rollout_worker.rollout_func.Stepping", FakeStepping
        )
        from malib.rollout.rollout_worker import RolloutWorker

        self.worker = RolloutWorker("test", env_desc, **kwargs)

    def test_actor_pool_checking(self):
        num_eval_actors = self.locals["kwargs"]["num_eval_actors"]
        num_rollout_actors = self.locals["kwargs"]["num_rollout_actors"]

        assert len(self.worker.actors) == num_eval_actors + num_rollout_actors

    # def test_simulation_exec(self):
    #     agent_ids = self.locals['env_desc']['possible_agents']
    #     observation_space = self.locals['env_desc']['observation_spaces'][agent_ids[0]]
    #     action_space = self.locals['env_desc']['action_spaces'][agent_ids[0]]
    #     task_desc = TaskDescription.gen_template(
    #         task_type=TaskType.SIMULATION,
    #         state_id="test_{}".format(time.time()),
    #         content={
    #             "policy_combinations": None,
    #             "agent_involve_info": {
    #                 "agent_ids": agent_ids,
    #                 "observation_space": observation_space,
    #                 "action_space": action_space,
    #             },
    #             "max_episode_length": 10,
    #         },
    #     )
    #     self.worker.simulation(task_desc)

    def test_rollout_exec(self):
        agent_ids = self.locals["env_desc"]["possible_agents"]
        observation_space = self.locals["env_desc"]["observation_spaces"][agent_ids[0]]
        action_space = self.locals["env_desc"]["action_spaces"][agent_ids[0]]
        task_desc = TaskDescription.gen_template(
            task_type=TaskType.ROLLOUT,
            state_id="test_{}".format(time.time()),
            content={
                "fragment_length": 10,
                "num_episodes": 1,
                "episode_seg": 1,
                "terminate_mode": None,
                "mode": None,
                "stopper": "simple_rollout",
                "stopper_config": {"max_step": 1},
                "agent_involve_info": {
                    "agent_ids": agent_ids,
                    "observation_space": observation_space,
                    "action_space": action_space,
                },
            },
        )
        self.worker.rollout(task_desc)
