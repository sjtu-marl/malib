from typing import List, Dict, Any

import pytest
import ray

from malib.common.strategy_spec import StrategySpec
from malib.rollout.inference import env_runner
from malib.rollout.inference.client import InferenceClient
from malib.rollout.envs import mdp
from malib.rollout.config import RolloutConfig
from malib.rl.random import RandomPolicy


@pytest.mark.parametrize(
    "env_desc,max_env_num",
    [
        [mdp.env_desc_gen(env_id="multi_round_nmdp"), 1],
    ],
)
def test_env_runner(env_desc: Dict[str, Any], max_env_num: int):
    # mapping from agents to agents
    with ray.init(local_mode=True):
        runner = env_runner.BasicEnvRunner(
            lambda: env_desc["creator"](**env_desc["config"]),
            max_env_num,
            use_subproc_env=False,
        )
        agents = env_desc["possible_agents"]
        observation_spaces = env_desc["observation_spaces"]
        action_spaces = env_desc["action_spaces"]

        inference_remote_cls = InferenceClient.as_remote(num_cpus=1)
        strategy_specs = {
            agent: StrategySpec(
                policy_cls=RandomPolicy,
                observation_space=observation_spaces["default"],
                action_space=action_spaces["default"],
                identifier=agent,
                policy_ids=["policy-0"],
            )
            for agent in agents
        }
        rollout_config = RolloutConfig(
            num_workers=1,
            eval_interval=1,
            n_envs_per_worker=10,
            use_subproc_env=False,
            timelimit=256,
        )

        infer_clients = {
            agent: inference_remote_cls.remote(
                model_entry_point=None,
                policy_cls=RandomPolicy,
                observation_space=observation_spaces[agent],
                action_space=action_spaces[agent],
                model_config=None,
            )
            for agent in agents
        }

        stats = runner.run(
            rollout_config,
            strategy_specs,
            inference_clients=infer_clients,
            data_entrypoints=None,
        )

        print(stats)
