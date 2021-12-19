import os
import pytest
import ray
import yaml

from pytest_mock import MockerFixture

from malib.agent.agent_interface import AgentFeedback
from malib.utils import logger
from malib.utils.typing import TaskRequest, TaskType
from malib.backend.coordinator.task import CoordinatorServer

from .mixin import ServerTestMixin


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
class TestServer(ServerTestMixin):
    def init_custom(self, exp_cfg, config):
        possible_agents = config["env_description"]["possible_agents"]
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

        # ray.get(self.remote_server.start.remote())

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
                CoordinatorServer, handler_name
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
            CoordinatorServer, f"_request_{name}", return_value=ret_value
        )
        mocked(self.server, None)

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
        logger.terminate()


# @pytest.mark.parametrize(
#     "c_path,use_policy_pool", [("examples/configs/mpe/ddpg_simple_nips.yaml", True)]
# )
# class TestServer:
#     @pytest.fixture(autouse=True)
#     def _init(self, c_path, use_policy_pool):
#         ray.init(local_mode=True)
#         with open(os.path.join(BASE_DIR, c_path), "r") as f:
#             config = yaml.safe_load(f)
#         global_configs = update_configs(config)
#         if global_configs["training"]["interface"].get("worker_config") is None:
#             global_configs["training"]["interface"]["worker_config"] = {
#                 "num_cpus": None,
#                 "num_gpus": None,
#                 "memory": None,
#                 "object_store_memory": None,
#                 "resources": None,
#             }
#         exp_cfg = logger.start(
#             group=global_configs.get("group", "experiment"),
#             name=global_configs.get("name", "case") + f"_{time.time()}",
#         )
#         self.use_policy_pool = use_policy_pool
#         self.server = CoordinatorServer.remote(
#             exp_cfg=exp_cfg, global_configs=global_configs
#         )

#     def test_server_launch(self):
#         ray.get(self.server.start.remote(use_init_policy_pool=self.use_policy_pool))
#         ray.shutdown()

#     def task_generation(self):
#         pass
