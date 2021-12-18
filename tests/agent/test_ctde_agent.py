import pytest
import ray
from ray.worker import time_string

from malib import settings
from malib.agent.agent_interface import AgentTaggedFeedback
from malib.agent.ctde_agent import CTDEAgent
from malib.algorithm.ddpg import CONFIG
from malib.utils.typing import BufferDescription, ParameterDescription

from tests.dataset import FakeDataServer
from tests.coordinator import FakeCoordinator
from tests.parameter_server import FakeParameterServer

from . import AgentTestMixin


@pytest.mark.parametrize(
    "agent_cls,yaml_path",
    [
        (CTDEAgent, "examples/configs/mpe/maddpg_simple_spread.yaml"),
    ],
)
class TestCTDE(AgentTestMixin):
    def init_coordinator(self):
        return FakeCoordinator.options(name=settings.COORDINATOR_SERVER_ACTOR).remote()

    def init_dataserver(self):
        return FakeDataServer.options(name=settings.OFFLINE_DATASET_ACTOR).remote()

    def init_parameter_server(self):
        return FakeParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR
        ).remote()

    def test_parameter_description_gen(self):
        env_aid = None
        policy_id = None
        trainable = None
        data = None

        desc: ParameterDescription = self.instance.parameter_desc_gen()
        assert desc.env_id == self.CONFIG["env_desc"]["config"]["env_id"]
        assert desc.identify == env_aid
        assert desc.id == policy_id
        assert desc.data == data
        assert desc.lock == (not trainable)

        pytest.fixture(scope="class", name="parameter_desc")(lambda: desc)

    def test_get_stationary_state(self):
        feedback: AgentTaggedFeedback = self.instance.get_stationary_state()

    def test_buffer_description_gen(self):
        batch_size = 64
        sample_mode = "time_step"

        self.instance.register
        agent_policy_mapping = None
        # CTDE agent should generate a single buffer description for all of its governed agents
        buffer_desc = self.instance.gen_buffer_description(
            agent_policy_mapping=None, batch_size=batch_size, sample_mode=sample_mode
        )
        assert isinstance(buffer_desc, BufferDescription), type(buffer_desc)
        # check keys in buffer desc
        assert (
            buffer_desc.env_id == self.CONFIGS["env_desc"]["config"]["env_id"]
        ), buffer_desc.env_id
        assert isinstance(buffer_desc.agent_id, (list, tuple)), type(
            buffer_desc.agent_id
        )
        # CTDE agent should include only one team, that is, all agents will be registered in _groups, then we can visit them
        #   via method agent_group()
        assert len(buffer_desc.agent_id) == len(self.instance.agent_group())
        for e in buffer_desc.agent_id:
            assert e in self.instance.agent_group()

        assert buffer_desc.batch_size == batch_size, buffer_desc.batch_size
        assert buffer_desc.sample_mode == sample_mode, buffer_desc.sample_mode

        pytest.fixture(scope="class", name="buffer_desc")(lambda: buffer_desc)

    def test_data_request(self, buffer_desc):
        res, size = self.instance.request_data(buffer_desc)

    def test_training(self, buffer_desc):
        # we use an easy training config here
        task_desc = ray.get(self.coordinator.gen_training_task.remote())
        self.instance.train(task_desc, training_config={})
