import os
import yaml
import pytest

from malib.settings import BASE_DIR
from malib.backend.datapool.parameter_server import ParameterServer
from malib.backend.datapool.offline_dataset_server import OfflineDataset
from malib.envs import get_env_cls
from malib.utils.general import update_configs


class AgentTestMixin:
    @pytest.fixture(autouse=True)
    def setup(self, agent_cls, yaml_path):
        # parse yaml configs
        with open(os.path.join(BASE_DIR, yaml_path), "r") as f:
            config = yaml.safe_load(f)
        self.CONFIGS = config
        self.dataset = self.init_dataserver()
        self.parameter_server = self.init_parameter_server()
        self.coordinator = self.init_coordinator_server()

        self.CONFIGS = update_configs(config)
        self.instance = agent_cls(
            assign_id="Learner",
            env_desc=self.CONFIGS["env_description"],
            algorithm_candidates=self.CONFIGS["algorithms"],
            training_agent_mapping=self.CONFIGS["agent_mapping_func"],
            **self.CONFIGS["training"]["interface"]
        )

        self.learner.start()

    def init_dataserver(self):
        raise NotImplementedError

    def init_parameter_server(self):
        raise NotImplementedError

    def init_coordinator(self):
        raise NotImplementedError

    def test_policies_getter(self):
        # return a dict of policies
        policies = self.instance.get_policies()
        assert isinstance(policies, dict)

    def test_register_env_agents(self):
        """Test whether environment agent register and agent_group interfaces"""

        self.instance.register_env_agent(self.governed_agents)
        assert len(self.instance.agent_group()) == len(self.governed_agents)
        for agent in self.governed_agents:
            assert agent in self.instance.agent_group()

    def test_policies_getter(self):
        # register policies with given algorithm candidates
        # and check the policy types
        policies = self.instance.policies()
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
