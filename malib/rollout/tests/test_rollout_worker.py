from types import LambdaType
from typing import Any, List, Dict
from collections import defaultdict
from unittest import mock

import time
import os
import pytest
import datetime
import random

import ray
import gym

from ray.util.queue import Queue

from malib import settings
from malib.utils.typing import AgentID
from malib.rollout.envs import dummy_env
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.random import RandomPolicy
from malib.backend.offline_dataset_server import OfflineDataset
from malib.backend.parameter_server import ParameterServer
from tests.coordinator import FakeCoordinator


@ray.remote(num_cpus=0)
class MockedInferenceClient:
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: LambdaType = None,
    ):
        pass

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these \
            new environment instances into it,otherwise do nothing.

        Args:
            maximum (int): Maximum limits.

        Returns:
            int: The number of nested environments.
        """

        return maximum

    def close(self):
        pass

    def run(
        self,
        agent_interfaces,
        desc: Dict[str, Any],
        dataserver_entrypoint: str = None,
        reset: bool = False,
    ) -> Dict[str, Any]:

        res = {
            "total_timesteps": desc["fragment_length"],
            "FPS": random.randrange(800, 1000),
        }

        if desc["flag"] in ["simulation", "evaluation"]:
            res["evaluation"] = {"episode_reward": random.random()}
        return res


@ray.remote(num_cpus=0)
class MockedInferenceServer:
    def __init__(
        self,
        agent_id: AgentID,
        observation_space: gym.Space,
        action_space: gym.Space,
        parameter_server: Any,
        governed_agents: List[AgentID],
        force_weight_update: bool = False,
    ) -> None:
        pass

    def shutdown(self):
        pass

    def save(self, model_dir: str) -> None:
        pass

    def connect(
        self,
        queues: List[Queue],
        runtime_config: Dict[str, Any],
        runtime_id: int,
    ):
        pass


@pytest.mark.parametrize("agent_mapping_func", [lambda agent: agent])
def test_rollout_worker(agent_mapping_func: LambdaType):
    timestamp = time.time()
    date = datetime.datetime.fromtimestamp(timestamp)
    experiment_tag = f"test_rollout_worker_{int(timestamp)}"
    log_dir = os.path.join(settings.LOG_DIR, "test_rollout_worker", str(date))

    if not ray.is_initialized():
        ray.init()
    c = FakeCoordinator.options(name=settings.COORDINATOR_SERVER_ACTOR).remote()
    d = OfflineDataset.options(name=settings.OFFLINE_DATASET_ACTOR).remote(100)
    p = ParameterServer.options(name=settings.PARAMETER_SERVER_ACTOR).remote()
    ray.get([c.start.remote(), d.start.remote(), p.start.remote()])

    from malib.rollout.pb_rolloutworker import RolloutWorker

    print("mock done")

    env_desc = dummy_env.env_desc_gen()

    rolloutworker = RolloutWorker(
        env_desc=env_desc,
        agent_mapping_func=agent_mapping_func,
        runtime_configs=dict(
            fragment_length=100,
            max_step=10,
            num_eval_episodes=10,
            num_threads=2,
            num_env_per_thread=1,
            num_eval_threads=1,
            use_subproc_env=False,
            batch_mode="time_step",
            postprocessor_types=["default"],
            eval_interval=3,
        ),
        log_dir=log_dir,
        experiment_tag=experiment_tag,
        reverb_table_kwargs=None,
        outer_inference_server=MockedInferenceServer,
        outer_inference_client=MockedInferenceClient,
    )

    runtime_obs_spaces = {}
    runtime_act_spaces = {}
    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]

    # map agents
    agent_group = defaultdict(lambda: [])
    runtime_agent_ids = []
    for agent in env_desc["possible_agents"]:
        runtime_id = agent_mapping_func(agent)
        agent_group[runtime_id].append(agent)
        runtime_agent_ids.append(runtime_id)
    runtime_agent_ids = set(runtime_agent_ids)
    agent_group = dict(agent_group)

    for rid, agents in agent_group.items():
        runtime_obs_spaces[rid] = obs_spaces[agents[0]]
        runtime_act_spaces[rid] = act_spaces[agents[0]]

    # step rollout
    runtime_strategy_specs = {
        rid: StrategySpec(
            identifier=rid,
            policy_ids=["dummy"],
            meta_data={
                "policy_cls": RandomPolicy,
                "experiment_tag": experiment_tag,
                "kwargs": {
                    "observation_space": runtime_obs_spaces[rid],
                    "action_space": runtime_act_spaces[rid],
                    "custom_config": {},
                    "model_config": {},
                },
            },
        )
        for rid in rolloutworker.runtime_agent_ids
    }
    stopping_conditions = {
        "reward_and_iteration": {"max_iteration": 10, "minimum_reward_improvement": 0.1}
    }
    print("stopping conditions done")
    rolloutworker.rollout(
        runtime_strategy_specs=runtime_strategy_specs,
        stopping_conditions=stopping_conditions,
    )
    print("rollout done")

    # step simulation
    runtime_strategy_specs_list = [runtime_strategy_specs for _ in range(5)]
    rolloutworker.simulate(runtime_strategy_specs_list=runtime_strategy_specs_list)
