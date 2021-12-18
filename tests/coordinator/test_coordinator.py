import os
import pytest
import ray
import yaml

from pytest_mock import MockerFixture

from malib import settings
from malib.agent.agent_interface import AgentFeedback
from malib.envs.gym import env_desc_gen
from malib.utils import logger
from malib.utils.typing import TaskRequest, TaskType
from malib.utils.general import update_configs
from malib.envs import gen_env_desc
from malib.backend.coordinator.task import CoordinatorServer

from tests.dataset import FakeDataServer
from tests.coordinator import FakeCoordinator
from tests.parameter_server import FakeParameterServer


@pytest.mark.parametrize(
    "yaml_path",
    [
        # marl case
        ("examples/configs/mpe/maddpg_simple_spread.yaml")
        # pbmarl case
        # ("examples/configs/")
    ],
    scope="class",
)
class TestServer:
    @pytest.fixture(autouse=True)
    def setup(self, yaml_path):
        # load yaml
        full_path = os.path.join(settings.BASE_DIR, yaml_path)
        assert os.path.exists(full_path), full_path
        with open(full_path, "r") as f:
            config = yaml.safe_load(f)

        config = update_configs(config)
        config["env_description"] = gen_env_desc(config["env_description"]["creator"])(
            **config["env_description"]["config"]
        )

        env_desc = config["env_description"]
        interface_config = config["training"]["interface"]
        interface_config["observation_spaces"] = env_desc["observation_spaces"]
        interface_config["action_spaces"] = env_desc["action_spaces"]
        self.task_mode = config["task_mode"]
        self.CONFIG = config

        exp_cfg = logger.start("test", "test")
        possible_agents = self.CONFIG["env_description"]["possible_agents"]
        self.server = CoordinatorServer(exp_cfg=exp_cfg, **config)
        self.request_template = TaskRequest(
            task_type=TaskType.OPTIMIZE,
            content=AgentFeedback(
                id="test",
                trainable_pairs={agent: "policy_0" for agent in possible_agents},
                statistics={},
            ),
            state_id="test",
        )

        self.init_remote_servers(exp_cfg, config)

    def init_remote_servers(self, exp_cfg, config):

        dataset = None

        try:
            dataset = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
        except ValueError:
            dataset = FakeDataServer.options(
                name=settings.OFFLINE_DATASET_ACTOR
            ).remote()

        parameter_server = None
        try:
            parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        except ValueError:
            parameter_server = FakeParameterServer.options(
                name=settings.PARAMETER_SERVER_ACTOR
            ).remote()

        self.dataset = dataset
        self.parameter_server = parameter_server

        self.remote_server = None

        try:
            self.remote_server = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)
        except ValueError:
            self.remote_server = (
                CoordinatorServer.as_remote()
                .options(name=settings.COORDINATOR_SERVER_ACTOR)
                .remote(exp_cfg=exp_cfg, **config)
            )
        ray.get(self.remote_server.start.remote())

    def test_task_register(self):
        for task_type in [
            TaskType.OPTIMIZE.value,
            TaskType.SIMULATION.value,
            TaskType.EVALUATE.value,
            TaskType.UPDATE_PAYOFFTABLE.value,
            TaskType.ROLLOUT.value,
        ]:
            handler_name = f"_request_{task_type}"
            assert getattr(
                self.server, handler_name
            ), f"`{handler_name}` registered failed in CoordinatorServer."

    @pytest.mark.parametrize(
        "name,ret_value",
        [
            (TaskType.OPTIMIZE.value, None),
            (TaskType.SIMULATION.value, None),
            (TaskType.EVALUATE.value, None),
            (TaskType.UPDATE_PAYOFFTABLE.value, None),
            (TaskType.ROLLOUT.value, None),
        ],
    )
    def test_handler(self, mocker: MockerFixture, name, ret_value):
        mocked = mocker.patch.object(
            self.server, f"_request_{name}", return_value=ret_value
        )
        assert mocked(self.request_template) == None

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
        logger.terminate()
