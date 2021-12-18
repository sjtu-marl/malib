import pytest


class AgentTestMixin:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        agent_cls,
        env_desc,
        algorithm_candidates,
        training_agent_mapping,
        observation_spaces,
        action_spaces,
        exp_cfg,
        use_init_policy_pool,
        population_size,
        algorithm_mapping,
        local_buffer_config,
    ):
        self.CONFIGS = locals()
        self.dataset = self.init_dataserver()
        self.parameter_server = self.init_parameter_server()
        self.coordinator = self.init_coordinator_server()

        self.instance = agent_cls(
            env_desc=env_desc,
            algorithm_candidates=algorithm_candidates,
            training_agent_mapping=training_agent_mapping,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            exp_cfg=exp_cfg,
            use_init_policy_pool=use_init_policy_pool,
            population_size=population_size,
            algorithm_mapping=algorithm_mapping,
            local_buffer_config=local_buffer_config,
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

        # got agents in group

        desc = self.instance.parameter_desc_gen()

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
