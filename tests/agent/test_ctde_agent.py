import pytest

from malib.agent.ctde_agent import CTDEAgent
from malib.utils.typing import BufferDescription

from tests.dataset import FakeDataServer
from tests.coordinator import FakeCoordinator
from tests.parameter_server import FakeParameterServer

from . import AgentTestMixin


@pytest.mark.parametrize(
    "agent_cls,env_desc,algorithm_candidates,training_agent_mapping,observation_spaces,action_spaces,exp_cfg,use_init_policy_pool,population_size,algorithm_mapping,local_buffer_config",
    [
        (CTDEAgent, ...),
    ],
)
class TestCTDE(AgentTestMixin):
    def init_coordinator(self):
        return FakeCoordinator.remote()

    def init_dataserver(self):
        return FakeDataServer.remote()

    def init_parameter_server(self):
        return FakeParameterServer.remote()

    def test_get_stationary_state(self):
        raise NotImplementedError

    def test_parameter_description_gen(self):
        return super().test_parameter_description_gen()

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

    def test_data_request(self):
        buffer_desc = self.instance.gen_buffer_description()
        self.instance.request_data(buffer_desc)

    def test_training(self):
        return super().test_training()
