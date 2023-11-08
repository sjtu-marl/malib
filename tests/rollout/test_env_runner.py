from typing import List, Dict, Any

import pytest

from malib.utils.typing import BehaviorMode
from malib.common.strategy_spec import StrategySpec
from malib.rollout.inference import env_runner
from malib.rollout.inference.client import InferenceClient
from malib.rollout.envs import mdp
from malib.rl.random import RandomPolicy


@pytest.mark.parametrize(
    "env_desc,max_env_num",
    [
        [mdp.env_desc_gen(env_id="multi_round_nmdp"), 1],
    ],
)
def test_env_runner(env_desc: Dict[str, Any], max_env_num: int):
    # mapping from agents to agents
    agent_groups = dict(zip(env_desc["possible_agents"], env_desc["possible_agents"]))
    runner = env_runner.EnvRunner(
        env_desc, max_env_num, agent_groups, use_subproc_env=False
    )

    agents = env_desc["possible_agents"]
    observation_spaces = env_desc["observation_spaces"]
    action_spaces = env_desc["action_spaces"]

    inference_remote_cls = InferenceClient.as_remote(num_cpus=1)
    rollout_config = {
        "flag": "evaluation",
        "strategy_specs": {
            agent: StrategySpec(
                policy_cls=RandomPolicy,
                observation_space=observation_spaces["default"],
                action_space=action_spaces["default"],
                identifier=agent,
                policy_ids=["policy-0"],
            )
            for agent in agents
        },
        "behavior_mode": BehaviorMode.EXPLOITATION,
    }

    infer_clients = {
        agent: inference_remote_cls.remote(
            agent, observation_spaces[agent], action_spaces[agent]
        )
        for agent in agents
    }

    runner.run(infer_clients, rollout_config)
