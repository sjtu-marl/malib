import os
import pytest
import yaml
import ray

from malib import settings
from malib.utils import logger
from malib.utils.general import update_configs
from malib.backend.coordinator.task import CoordinatorServer

from tests.dataset import FakeDataServer
from tests.parameter_server import FakeParameterServer


class ServerTestMixin:
    @pytest.fixture(autouse=True)
    def setup_class(self, yaml_path):
        # load yaml
        full_path = os.path.join(settings.BASE_DIR, yaml_path)
        assert os.path.exists(full_path), full_path
        with open(full_path, "r") as f:
            config = yaml.safe_load(f)
        config = update_configs(config)
        self.task_mode = config["task_mode"]
        self.CONFIG = config

        exp_cfg = logger.start("test", "test")
        self.init_remote_servers(exp_cfg, config)
        self.init_custom(exp_cfg, config)

    def init_custom(self, exp_cfg, config):
        raise NotImplementedError

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

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
        if logger.logger_server is not None:
            logger.terminate()
