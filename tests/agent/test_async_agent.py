import pytest
import ray

from pytest_mock import MockerFixture

from malib.agent.agent_interface import AgentTaggedFeedback
from malib.agent.async_agent import AsyncAgent
from malib.algorithm.common.trainer import Trainer
from malib.utils.typing import BufferDescription, ParameterDescription, Dict, Any

from . import AgentTestMixin


@pytest.mark.parametrize(
    "agent_cls,yaml_path",
    [
        (AsyncAgent, "examples/configs/mpe/maddpg_simple_spread.yaml"),
    ],
    scope="class",
)
class TestAsyncAgent(AgentTestMixin):
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
        assert isinstance(buffer_desc, BufferDescription), type(buffer_desc)
        # check keys in buffer desc
        assert (
            buffer_desc.env_id == self.CONFIGS["env_description"]["config"]["env_id"]
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

    def test_parameter_push_and_pull(self):
        pass

    def test_optimize(self, mocker: MockerFixture):
        agent_policy_mapping = {k: v[0] for k, v in self.trainable_pairs.items()}
        batch = {agent: {} for agent in self.governed_agents}

        class faketrainer(Trainer):
            def reset(self, policy, training_config):
                pass

            def preprocess(self, batch, **kwargs):
                return batch

            def optimize(self, batch) -> Dict[str, Any]:
                return {"ploss": 0.0, "vloss": 0.0, "gradients": 0.0}

        self.instance._trainers = {k: faketrainer(k) for k in self.instance._trainers}
        res = self.instance.optimize(agent_policy_mapping, batch, training_config={})
        assert isinstance(res, dict), res
        for k, v in res.items():
            assert isinstance(v, float)

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
