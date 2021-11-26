import importlib
import pytest

from malib.utils.typing import BufferDescription
from malib.envs import Environment
from malib.envs.tests import build_dummy_agent_interfaces
from malib.envs.vector_env import VectorEnv
from malib.rollout.rollout_func import (
    env_runner,
    _do_policy_eval,
    _process_policy_outputs,
    _process_environment_returns,
)
from malib.backend.datapool.test import FakeDataServer


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}),
        ("malib.envs.mpe", "MPE", "simple_spread_v2", {"max_cycles": 25}),
    ],
)
class TestEnvRunner:
    @pytest.fixture(autouse=True)
    def _init(self, module_path, cname, env_id, scenario_configs):
        creator = getattr(importlib.import_module(module_path), cname)
        env: Environment = creator(env_id=env_id, scenario_configs=scenario_configs)

        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces

        vec_env = VectorEnv(
            observation_spaces,
            action_spaces,
            creator,
            configs={"scenario_configs": scenario_configs, "env_id": env_id},
        )

        agent_interfaces = build_dummy_agent_interfaces(
            observation_spaces, action_spaces
        )

        self.vec_env = vec_env
        self.agent_interfaces = agent_interfaces

        self.vec_env.add_envs(num=4)

    def test_process_in_runner(self):
        runtime_config = {
            "num_envs": 2,
            "fragment_length": 100,
            "max_step": 25,
            "custom_reset_config": None,
        }
        _ = [interface.reset() for interface in self.agent_interfaces.values()]
        rets = self.vec_env.reset(
            limits=runtime_config["num_envs"],
            fragment_length=runtime_config["fragment_length"],
            max_step=runtime_config["max_step"],
            custom_reset_config=runtime_config["custom_reset_config"],
        )

        while not self.vec_env.is_terminated():
            policy_inputs, filtered_ouptuts = _process_environment_returns(
                env_rets=rets,
                agent_interfaces=self.agent_interfaces,
            )

            # check consistency in env ids
            pinput_env_ids = sorted(list(policy_inputs.keys()))
            filtered_env_ids = sorted(list(filtered_ouptuts.keys()))
            real_env_ids = sorted(list(self.vec_env.active_envs.keys()))

            assert pinput_env_ids == filtered_env_ids == real_env_ids, (
                pinput_env_ids,
                filtered_env_ids,
                real_env_ids,
            )

            policy_outputs, active_env_ids = _do_policy_eval(
                policy_inputs, self.agent_interfaces
            )
            sorted_active_env_ids = sorted(active_env_ids)
            assert sorted_active_env_ids == pinput_env_ids

            env_inputs, detached_policy_outputs = _process_policy_outputs(
                active_env_ids, policy_outputs, self.vec_env
            )

            detached_env_ids = sorted(detached_policy_outputs.keys())
            assert sorted_active_env_ids == detached_env_ids, (
                sorted_active_env_ids,
                detached_env_ids,
            )

            rets = self.vec_env.step(env_inputs)

    def test_env_runner_no_buffer_send(self):
        # random select agent behavior policies
        _ = [interface.reset() for interface in self.agent_interfaces.values()]
        behavior_policy_ids = {
            agent_id: interface.behavior_policy
            for agent_id, interface in self.agent_interfaces.items()
        }
        rollout_info = env_runner(
            self.vec_env,
            self.agent_interfaces,
            buffer_desc=None,
            runtime_config={
                "max_step": 10,
                "num_envs": 2,
                "fragment_length": 100,
                "behavior_policies": behavior_policy_ids,
                "custom_reset_config": None,
            },
            dataset_server=None,
        )

    def test_env_runner_with_buffer_send(self):
        _ = [interface.reset() for interface in self.agent_interfaces.values()]
        behavior_policy_ids = {
            agent_id: interface.behavior_policy
            for agent_id, interface in self.agent_interfaces.items()
        }
        dataset = FakeDataServer.remote()
        buffer_desc = BufferDescription(
            env_id=self.vec_env.env_configs["env_id"],
            agent_id=self.vec_env.possible_agents,
            policy_id=[
                behavior_policy_ids[aid] for aid in self.vec_env.possible_agents
            ],
        )

        rollout_info = env_runner(
            self.vec_env,
            self.agent_interfaces,
            buffer_desc=buffer_desc,
            runtime_config={
                "max_step": 10,
                "num_envs": 2,
                "fragment_length": 100,
                "behavior_policies": behavior_policy_ids,
                "custom_reset_config": None,
            },
            dataset_server=dataset,
        )
