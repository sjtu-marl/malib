import pytest

from malib.rollout.envs.dummy_env import env_desc_gen, DummyEnv
from malib.agent.indepdent_agent import IndependentAgent


@pytest.mark.parametrize("env_id", [])
class TestIndependentAgent:
    @pytest.fixture(autouse=True)
    def init(self, env_id: str):
        self.env_desc = env_desc_gen(num_agents=4, enable_env_state=False)

    def test_policy_adding(self):
        pass

    def test_parameter_sync(self):
        pass

    def test_training_logic(self):
        experiment_tag = "test_"
        agent_mapping_func = lambda agent: agent

        learners = {
            agent: IndependentAgent(
                experiment_tag=experiment_tag,
                runtime_id=agent,
                env_desc=self.env_desc,
                algorithms=None,
                agent_mapping_func=agent_mapping_func,
                governed_agents=[agent],
                trainer_config=None,
                custom_config=None,
                local_buffer_config=None,
            )
            for agent in self.env_desc["possible_agents"]
        }

        # then connect to remote server
        for learner in learners.values():
            learner.connect()

        # add policy
        for learner in learners.values():
            learner.add_policies(n=1)

        # check push and pull
        learners
