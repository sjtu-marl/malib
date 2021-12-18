import os
import time

import yaml
import pytest
import ray

from malib import settings
from malib.agent.agent_interface import AgentInterface
from malib.envs.gym import env_desc_gen
from malib.settings import BASE_DIR
from malib.backend.datapool.parameter_server import ParameterServer
from malib.backend.datapool.offline_dataset_server import OfflineDataset
from malib.envs import gen_env_desc
from malib.utils.general import update_configs
from malib.utils import logger
from malib.utils.typing import (
    TaskDescription,
    TaskType,
    TrainingDescription,
    AgentInvolveInfo,
)

from tests.dataset import FakeDataServer
from tests.coordinator import FakeCoordinator
from tests.parameter_server import FakeParameterServer


class AgentTestMixin:
    @pytest.fixture(autouse=True)
    def init(self, agent_cls, yaml_path):
        # parse yaml configs
        with open(os.path.join(BASE_DIR, yaml_path), "r") as f:
            config = yaml.safe_load(f)

        self.CONFIGS = config
        self.dataset = self.init_dataserver()
        self.parameter_server = self.init_parameter_server()
        self.coordinator = self.init_coordinator()

        self.CONFIGS = update_configs(config)

        self.CONFIGS["env_description"] = gen_env_desc(
            self.CONFIGS["env_description"]["creator"]
        )(**self.CONFIGS["env_description"]["config"])
        env_desc = self.CONFIGS["env_description"]

        interface_config = self.CONFIGS["training"]["interface"]

        exp_cfg = logger.start("test", "test")

        self.instance: AgentInterface = agent_cls(
            assign_id="Learner",
            env_desc=self.CONFIGS["env_description"],
            algorithm_candidates=self.CONFIGS["algorithms"],
            training_agent_mapping=self.CONFIGS["agent_mapping_func"],
            observation_spaces=env_desc["observation_spaces"],
            action_spaces=env_desc["action_spaces"],
            exp_cfg=exp_cfg,
            use_init_policy_pool=interface_config["use_init_policy_pool"],
            population_size=interface_config["population_size"],
            algorithm_mapping=interface_config["algorithm_mapping"],
        )

        self.instance.register_env_agent(
            self.CONFIGS["env_description"]["possible_agents"]
        )

        self.instance.start()
        task_request = self.instance.add_policy(
            TaskDescription(
                task_type=TaskType.ADD_POLICY,
                content=TrainingDescription(
                    agent_involve_info=AgentInvolveInfo(
                        training_handler="Learner",
                        trainable_pairs=dict.fromkeys(
                            self.CONFIGS["env_description"]["possible_agents"], None
                        ),
                        populations={},
                    ),
                ),
                state_id=str(time.time()),
            )
        )
        self.trainable_pairs = task_request.content.trainable_pairs

        self.governed_agents = self.CONFIGS["env_description"]["possible_agents"]

    def init_coordinator(self):
        server = None

        try:
            server = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)
        except ValueError:
            server = FakeCoordinator.options(
                name=settings.COORDINATOR_SERVER_ACTOR
            ).remote()

        return server

    def init_dataserver(self):
        server = None

        try:
            server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
        except ValueError:
            server = FakeDataServer.options(
                name=settings.OFFLINE_DATASET_ACTOR
            ).remote()

        return server

    def init_parameter_server(self):
        server = None
        try:
            server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        except ValueError:
            server = FakeParameterServer.options(
                name=settings.PARAMETER_SERVER_ACTOR
            ).remote()
        return server

    def test_policies_getter(self):
        # return a dict of policies
        policies = self.instance.get_policies()
        assert isinstance(policies, dict)

    def test_parameter_description_gen(self):
        """Test parameter desciption generator"""

        raise NotImplementedError

    def test_get_stationary_state(self):
        raise NotImplementedError

    def test_parameter_push_and_pull(self):
        """Test parameter push and pull"""

        raise NotImplementedError

    def test_data_request(self):
        """Test data request"""

        raise NotImplementedError

    def test_buffer_description_gen(self):
        """Test buffer description generation"""

        raise NotImplementedError

    def test_training(self):
        raise NotImplementedError

    def test_save(self):
        self.instance.save(None)

    def test_load(self):
        self.instance.load(None)

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
        logger.terminate()
