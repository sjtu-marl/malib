# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Type, Dict, Any
from collections import namedtuple, defaultdict

import pytest
import ray

from pytest_mock import MockerFixture
from gym import spaces

from malib import rl
from malib.common.strategy_spec import StrategySpec
from malib.scenarios import scenario, marl_scenario, psro_scenario

from malib.mocker.mocker_utils import (
    use_ray_env,
    FakeRolloutManager,
    FakeTrainingManager,
    FakePayoffManager,
)


nash_value = namedtuple("nash_value", "nash_conv")


@pytest.fixture
def env_desc():
    agents = [f"agent_{i}" for i in range(3)]
    return {
        "creator": lambda x: print("create an fake enviornment"),
        "possible_agents": agents,
        "observation_spaces": {
            agent: spaces.Box(-1.0, 1.0, shape=(2,)) for agent in agents
        },
        "action_spaces": {agent: spaces.Discrete(2) for agent in agents},
        "config": {"env_id": "fake_env"},
    }


@pytest.mark.parametrize("rl_module", [rl.dqn, rl.pg])
def test_marl_scenario(
    mocker: MockerFixture, rl_module: Type, env_desc: Dict[str, Any]
):
    mocker.patch(
        "malib.scenarios.marl_scenario.RolloutWorkerManager", new=FakeRolloutManager
    )
    mocker.patch(
        "malib.scenarios.marl_scenario.TrainingManager", new=FakeTrainingManager
    )

    # run the gym example here
    training_config = rl_module.DEFAULT_CONFIG["training_config"].copy()
    algorithms = {"default": (rl_module.POLICY, rl_module.TRAINER, {}, {})}
    rollout_config = {}
    agent_mapping_func = lambda agent: agent

    scenario = marl_scenario.MARLScenario(
        name="test_marl_scenario",
        log_dir="./logs",
        env_description=env_desc,
        algorithms=algorithms,
        training_config=training_config,
        rollout_config=rollout_config,
        agent_mapping_func=agent_mapping_func,
        stopping_conditions={
            "training": {"max_iteration": int(1e10)},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )

    copy_scenario = scenario.copy()
    assert copy_scenario != scenario
    copy_scenario = scenario.with_updates(name="xxxx")
    assert copy_scenario != scenario
    assert copy_scenario.name == "xxxx"

    marl_scenario.execution_plan(experiment_tag="test_marl_scenario", scenario=scenario)


@pytest.mark.parametrize("rl_module", [rl.dqn, rl.pg])
def test_psro_scenario(
    mocker: MockerFixture, rl_module: Type, env_desc: Dict[str, Any]
):
    # pack simulation
    mocker.patch(
        "malib.scenarios.psro_scenario.TrainingManager", new=FakeTrainingManager
    )
    mocker.patch(
        "malib.scenarios.psro_scenario.RolloutWorkerManager", new=FakeRolloutManager
    )
    mocker.patch("malib.scenarios.psro_scenario.PayoffManager", new=FakePayoffManager)

    agent_mapping_func = lambda agent: agent
    agent_groups = defaultdict(lambda: set())
    for agent in env_desc["possible_agents"]:
        rid = agent_mapping_func(agent)
        agent_groups[rid].add(agent)
    rids = list(agent_groups.keys())
    strategy_specs = {
        rid: StrategySpec(
            rid,
            ["policy-0"],
            {
                "policy_cls": rl_module.POLICY,
                "kwargs": {
                    "observation_space": env_desc["observation_spaces"]["agent_0"],
                    "action_space": env_desc["action_spaces"]["agent_0"],
                    "model_config": rl_module.DEFAULT_CONFIG["model_config"],
                    "custom_config": {},
                    "kwargs": {},
                },
                "experiment_tag": "test_psro_scenario",
                "prob_list": [1],
            },
        )
        for rid in rids
    }
    eval_results = {}

    mocker.patch(
        "malib.scenarios.psro_scenario.marl_execution_plan",
        return_value={"strategy_specs": strategy_specs},
    )
    mocker.patch(
        "malib.scenarios.psro_scenario.measure_exploitability",
        return_value=nash_value(1.0),
    )
    mocker.patch(
        "malib.scenarios.psro_scenario.simulate",
        return_value=[(strategy_specs, eval_results)],
    )

    training_config = rl_module.DEFAULT_CONFIG["training_config"].copy()
    algorithms = {"default": (rl_module.POLICY, rl_module.TRAINER, {}, {})}
    rollout_config = {}
    agent_mapping_func = lambda agent: agent

    with use_ray_env() as (parameter_server, dataset_server):
        # upload parameters to remote parameter server
        ray.get(
            [
                parameter_server.create_table.remote(strategy_spec)
                for strategy_spec in strategy_specs.values()
            ]
        )
        for spec in strategy_specs.values():
            spec_id = spec.id
            for pid in spec.policy_ids:
                policy = spec.gen_policy()
                ray.get(
                    parameter_server.set_weights.remote(
                        spec_id, pid, policy.state_dict()
                    )
                )

        scenario = psro_scenario.PSROScenario(
            name="test_psro_scenario",
            log_dir="./logs",
            algorithms=algorithms,
            env_description=env_desc,
            training_config=training_config,
            rollout_config=rollout_config,
            # control the outer loop.
            global_stopping_conditions={"max_iteration": 1},
            agent_mapping_func=agent_mapping_func,
            # for the training of best response.
            stopping_conditions={
                "training": {"max_iteration": 1},
                "rollout": {"max_iteration": 1},
            },
        )
        scenario.parameter_server = parameter_server

        psro_scenario.execution_plan("test_psro_scenario", scenario)
