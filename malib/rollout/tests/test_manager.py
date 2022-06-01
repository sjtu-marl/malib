from typing import Dict, Any, List

import pytest
import ray

from malib.utils.typing import PolicyID
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.random import RandomPolicy
from malib.rollout.manager import RolloutWorkerManager
from malib.rollout.envs.gym import env_desc_gen
from malib.runner import start_servers


def gen_rollout_tasks(
    manager: RolloutWorkerManager,
    policy_ids: List[PolicyID] = [],
    prob_list: List[float] = None,
):
    runtime_ids = manager.runtime_ids
    # strategy spec for rollout cannot be empty.
    agent_groups = manager.agent_groups
    observation_spaces = manager.observation_spaces
    env_agents = list(observation_spaces.keys())
    action_spaces = manager.action_spaces
    selected_observation_spaces = {
        rid: observation_spaces[list(agent_set)[0]]
        for rid, agent_set in agent_groups.items()
    }
    selected_action_spaces = {
        rid: action_spaces[list(agent_set)[0]]
        for rid, agent_set in agent_groups.items()
    }

    task_list = [
        {
            "strategy_specs": {
                rid: StrategySpec(
                    identifier=rid,
                    policy_ids=policy_ids,
                    meta_data={
                        "policy_cls": RandomPolicy,
                        "prob_list": prob_list,
                        "experiment_tag": manager.experiment_tag,
                        "kwargs": {
                            "observation_space": selected_observation_spaces[rid],
                            "action_space": selected_action_spaces[rid],
                            "model_config": {},
                            "custom_config": {},
                            "kwargs": {},
                        },
                    },
                )
                for rid in runtime_ids
            },
            "tranable_agents": env_agents,
        }
    ]

    return task_list


@pytest.mark.parametrize(
    "env_desc,rollout_config",
    [
        (
            env_desc_gen(env_id="CartPole-v1", scenario_configs={}),
            {
                "fragment_length": 100,
                "max_step": 10,
                "num_eval_episodes": 5,
                "num_threads": 1,
                "num_env_per_thread": 2,
                "num_eval_threads": 1,
                "use_subproc_env": False,
                "batch_mode": "time_step",
                "postprocessor_types": ["defaults"],
                "eval_interval": 1,
            },
        )
    ],
)
class TestRolloutManager:
    @pytest.fixture(autouse=True)
    def init(self, rollout_config: Dict[str, Any], env_desc: Dict[str, Any]):
        resource_config = None
        log_dir = "/tmp/malib/test_rollout_manager"
        agent_mapping_func = lambda agent: agent

        if not ray.is_initialized():
            ray.init()

        start_servers()

        self.manager = RolloutWorkerManager(
            experiment_tag="test_rollout_manager",
            stopping_conditions={
                "reward_and_iteration": {
                    "max_iteration": 10,
                    "minimum_reward_improvement": 1.0,
                }
            },
            num_worker=1,
            agent_mapping_func=agent_mapping_func,
            rollout_config=rollout_config,
            env_desc=env_desc,
            log_dir=log_dir,
            resource_config=resource_config,
        )

    def test_rollout_with_value_error(self):
        task_list = gen_rollout_tasks(self.manager)
        with pytest.raises(ValueError):
            self.manager.rollout(task_list)

    def test_rollout_without_value_error(self):
        policy_ids = [f"policy-{i}" for i in range(3)]
        prob_list = [0.1, 0.2, 0.7]
        task_list = gen_rollout_tasks(self.manager, policy_ids, prob_list)
        self.manager.rollout(task_list)
        self.manager.wait()

    def test_simulate_task_dispatching(self):
        pass
