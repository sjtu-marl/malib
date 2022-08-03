import time

from collections import defaultdict

import pytest
import ray

from malib import settings
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.random import RandomPolicy
from malib.rollout.envs import dummy_env
from malib.rollout.inference.pipe.client import InferenceClient
from malib.rollout.inference.pipe.server import InferenceWorkerSet
from malib.backend.offline_dataset_server import OfflineDataset
from malib.backend.parameter_server import ParameterServer


@pytest.mark.parametrize("max_env_num", [1, 2])
def test_inference_coordination(max_env_num: int):
    """Test infrence coordination with Dummy environment. The coordination concentrates on data collection, broadcasting \
        and action compute with centralized inference server.
    """

    # Not determined yet:
    #   1. different agent mapping func
    #   2. remote inference client (we create this case with local client)

    env_desc = dummy_env.env_desc_gen()
    agent_map_func = lambda agent: agent
    tmp_env = env_desc["creator"](**env_desc["config"])
    env_agents = tmp_env.possible_agents

    runtime_ids = set([agent_map_func(aid) for aid in tmp_env.possible_agents])

    if not ray.is_initialized():
        ray.init()

    try:
        offline_dataset_server = (
            OfflineDataset.as_remote(num_cpus=0)
            .options(name=settings.OFFLINE_DATASET_ACTOR)
            .remote(table_capacity=100)
        )
    except ValueError:
        print("detected existing offline dataset server")

    try:
        parameter_server = (
            ParameterServer.as_remote(num_cpus=1)
            .options(name=settings.PARAMETER_SERVER_ACTOR)
            .remote()
        )
    except ValueError:
        print("detected exisitng parameter server")

    offline_dataset_server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
    parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)

    client = InferenceClient(
        env_desc=env_desc,
        dataset_server=offline_dataset_server,
        max_env_num=max_env_num,
        use_subproc_env=False,
        batch_mode="time_step",
        postprocessor_types=["defaults"],
        training_agent_mapping=agent_map_func,
    )

    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]

    runtime_obs_spaces = {}
    runtime_act_spaces = {}

    # map agents
    agent_group = defaultdict(lambda: [])
    runtime_agent_ids = []
    for agent in env_agents:
        runtime_id = agent_map_func(agent)
        agent_group[runtime_id].append(agent)
        runtime_agent_ids.append(runtime_id)
    runtime_agent_ids = set(runtime_agent_ids)
    agent_group = dict(agent_group)

    for rid, agents in agent_group.items():
        runtime_obs_spaces[rid] = obs_spaces[agents[0]]
        runtime_act_spaces[rid] = act_spaces[agents[0]]

    servers = {
        runtime_id: InferenceWorkerSet.remote(
            agent_id=runtime_id,
            observation_space=runtime_obs_spaces[runtime_id],
            action_space=runtime_act_spaces[runtime_id],
            parameter_server=parameter_server,
            governed_agents=agent_group[runtime_id],
        )
        for runtime_id in runtime_ids
    }

    policy_ids = ["dummy_0"]
    prob_list = [1.0]

    experiment_tag = f"test_inference_coordination_{time.time()}"
    dataserver_entrypoint = experiment_tag

    strategy_specs = {
        runtime_id: StrategySpec(
            identifier=runtime_id,
            policy_ids=policy_ids,
            meta_data={
                "prob_list": prob_list,
                "experiment_tag": experiment_tag,
                "policy_cls": RandomPolicy,
                "kwargs": {
                    "registered_name": "random",
                    "observation_space": runtime_obs_spaces[runtime_id],
                    "action_space": runtime_act_spaces[runtime_id],
                    "model_config": {},
                    "custom_config": {},
                    "kwargs": {},
                },
            },
        )
        for runtime_id in runtime_agent_ids
    }

    trainable_agents = env_agents
    # create table
    offline_dataset_server.create_table.remote(name=dataserver_entrypoint)

    client.run(
        agent_interfaces=servers,
        desc=dict(
            flag="rollout",
            strategy_specs=strategy_specs,
            trainable_agents=trainable_agents,
            agent_group=agent_group,
            fragment_length=100,
            max_step=10,
        ),
        dataserver_entrypoint=dataserver_entrypoint,
    )


def test_inference_server():
    # Things not be determined yet:
    #   1. validate the correctness of strategyspec's behavior
    pass
