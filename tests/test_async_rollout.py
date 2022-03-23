import pytest
import ray
import yaml
import os
import time

from malib import settings
from malib.utils.typing import AgentID, Dict, BufferDescription, ParameterDescription
from malib.envs.gr_football import env_desc_gen
from malib.rollout.inference_client import InferenceClient
from malib.rollout.inference_server import InferenceWorkerSet
from malib.backend.datapool.offline_dataset_server import OfflineDataset
from malib.backend.datapool.parameter_server import ParameterServer
from malib.algorithm.random import RandomPolicy


@pytest.fixture(scope="session")
def config():
    if not ray.is_initialized():
        ray.init(local_mode=True)
    yaml_path = os.path.join(
        settings.BASE_DIR, "examples/mappo_gfootball/mappo_5_vs_5.yaml"
    )

    with open(yaml_path, "w") as f:
        yaml_config = yaml.load(f)

    dataset_server = OfflineDataset()
    parameter_server = ParameterServer()

    _config = {
        "yaml": yaml_config,
        "parameter_server": parameter_server,
        "dataset_server": dataset_server,
    }

    return _config


def test_async_rollout(config):
    yaml_config = config["yaml"]
    parameter_server = config["parameter_server"]
    dataset_server = config["dataset_server"]

    env_desc = env_desc_gen(yaml_config["env_description"])
    postprocessors = yaml_config["rollout"]["postprocessor_types"]
    agents = env_desc["possible_agents"]

    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]

    client = InferenceClient.remote(
        env_desc,
        dataset_server=dataset_server,
        use_subproc_env=False,
        batch_mode=yaml_config["rollout"]["batch_mode"],
        postprocessor_types=postprocessors,
    )

    servers = {
        agent: InferenceWorkerSet.remote(
            agent_id=agent,
            observation_space=obs_spaces[agent],
            action_space=act_spaces[agent],
            parameter_server=parameter_server,
            force_weight_update=True,
        )
        for agent in agents
    }

    policy_template = RandomPolicy(
        "random",
        observation_space=obs_spaces["team_0"],
        action_space=act_spaces["team_0"],
        model_config=None,
        custom_config=None,
    )

    p_descs: Dict[AgentID, ParameterDescription] = {
        agent: ParameterDescription(
            time_stamp=time.time(),
            identify=agent,
            env_id=env_desc["config"]["env_id"],
            id="MAPPO_0",
            type=ParameterDescription.Type.PARAMETER,
            lock=False,
            description=policy_template.description,
            data=policy_template.state_dict(),
            parallel_num=1,
            version=0,
        )
        for agent in agents
    }

    _ = ray.get([parameter_server.push.remote(e) for e in p_descs.values()])

    max_step = 300
    num_episodes = 3
    trainable_pairs = {aid: "MAPPO_0" for aid in agents}
    parameter_server.push()
    task_desc = {
        "max_step": max_step,
        "num_episodes": num_episodes,
        "flag": "rollout",
        "behavior_policies": trainable_pairs,
        "paramter_desc_dict": p_descs,
    }
    fragment_length = max_step * num_episodes

    buffer_desc = BufferDescription(
        env_id=env_desc["config"]["env_id"],
        agent_id=list(trainable_pairs.keys()),
        policy_id=[pid for pid, _ in trainable_pairs.values()],
        capacity=None,
        sample_start_size=None,
    )

    ray.get(dataset_server.create_table.remote(buffer_desc))

    _ = ray.get(
        client.run.remote(
            servers,
            fragment_length=fragment_length,
            desc=task_desc,
            buffer_desc=buffer_desc,
        )
    )
