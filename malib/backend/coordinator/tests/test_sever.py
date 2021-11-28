import os
import yaml
import pytest
import time
import ray

from malib.settings import BASE_DIR
from malib.utils import logger
from malib.utils.general import update_configs
from malib.backend.coordinator.server import CoordinatorServer


@pytest.mark.parametrize(
    "c_path,use_policy_pool", [("examples/configs/mpe/ddpg_simple_nips.yaml", True)]
)
class TestServer:
    @pytest.fixture(autouse=True)
    def _init(self, c_path, use_policy_pool):
        ray.init(local_mode=True)
        with open(os.path.join(BASE_DIR, c_path), "r") as f:
            config = yaml.safe_load(f)
        global_configs = update_configs(config)
        if global_configs["training"]["interface"].get("worker_config") is None:
            global_configs["training"]["interface"]["worker_config"] = {
                "num_cpus": None,
                "num_gpus": None,
                "memory": None,
                "object_store_memory": None,
                "resources": None,
            }
        exp_cfg = logger.start(
            group=global_configs.get("group", "experiment"),
            name=global_configs.get("name", "case") + f"_{time.time()}",
        )
        self.use_policy_pool = use_policy_pool
        self.server = CoordinatorServer.remote(
            exp_cfg=exp_cfg, global_configs=global_configs
        )

    def test_server_launch(self):
        ray.get(self.server.start.remote(use_init_policy_pool=self.use_policy_pool))
        ray.shutdown()

    def task_generation(self):
        pass
