import pytest
import ray

from malib.agent.agent_interface import AgentTaggedFeedback
from malib.agent.sync_agent import SyncAgent
from malib.utils.typing import AgentID, BufferDescription, ParameterDescription

from . import AgentTestMixin


@pytest.mark.parametrize(
    "agent_cls,yaml_path",
    [
        (SyncAgent, "examples/configs/mpe/sync_mode_ddpg_simple_spread.yaml"),
    ],
    scope="class",
)
class TestSyncAgent(AgentTestMixin):
    def test_parameter_description_gen(self):
        agent_policy_mapping = {k: v[0] for k, v in self.trainable_pairs.items()}
        env_aid = list(agent_policy_mapping.keys())[0]
        policy_id = list(agent_policy_mapping.values())[0]
        trainable = False
        data = None

        desc: ParameterDescription = self.instance.parameter_desc_gen(
            env_aid, policy_id, trainable, data
        )
        assert desc.env_id == self.CONFIGS["env_description"]["config"]["env_id"]
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

        agent_policy_mapping = {k: v[0] for k, v in self.trainable_pairs.items()}
        # CTDE agent should generate a single buffer description for all of its governed agents
        buffer_desc = self.instance.gen_buffer_description(
            agent_policy_mapping=agent_policy_mapping,
            batch_size=batch_size,
            sample_mode=sample_mode,
        )
        assert isinstance(buffer_desc, dict), type(buffer_desc)
        # check keys in buffer desc

        for k, v in buffer_desc.items():
            assert k in self.instance.agent_group()
            assert (
                v.env_id == self.CONFIGS["env_description"]["config"]["env_id"]
            ), v.env_id
            assert isinstance(v.agent_id, AgentID), type(v.agent_id)
            assert v.batch_size == batch_size, v.batch_size
            assert v.sample_mode == sample_mode, v.sample_mode

        assert len(buffer_desc) == len(self.instance.agent_group())

        pytest.fixture(scope="class", name="buffer_desc")(lambda: buffer_desc)

    def test_parameter_push_and_pull(self):
        pass

    def test_data_request(self):
        batch_size = 64
        sample_mode = "time_step"

        agent_policy_mapping = {k: v[0] for k, v in self.trainable_pairs.items()}
        buffer_desc = self.instance.gen_buffer_description(
            agent_policy_mapping=agent_policy_mapping,
            batch_size=batch_size,
            sample_mode=sample_mode,
        )
        res, size = self.instance.request_data(buffer_desc)

    def test_training(self):
        batch_size = 64
        sample_mode = "time_step"

        agent_policy_mapping = {k: v[0] for k, v in self.trainable_pairs.items()}
        buffer_desc = self.instance.gen_buffer_description(
            agent_policy_mapping=agent_policy_mapping,
            batch_size=batch_size,
            sample_mode=sample_mode,
        )
        # we use an easy training config here
        task_desc = ray.get(self.coordinator.gen_training_task.remote())
        # self.instance.train(task_desc, training_config={})
