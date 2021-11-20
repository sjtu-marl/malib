import logging
import ray
import pytest
import time
import yaml
import os
import copy

from malib import settings
from malib.utils import logger
from malib.utils.preprocessor import get_preprocessor

from malib.backend.datapool.parameter_server import ParameterServer
from malib.backend.datapool.offline_dataset_server import Episode, OfflineDataset
from malib.backend.coordinator.light_server import LightCoordinator
from malib.envs.gym.env import GymEnv


def update_configs(update_dict, ori_dict=None):
    """Update global configs with a given dict"""

    ori_configs = (
        copy.copy(ori_dict)
        if ori_dict is not None
        else copy.copy(settings.DEFAULT_CONFIG)
    )

    for k, v in update_dict.items():
        # assert k in ori_configs, f"Illegal key: {k}, {list(ori_configs.keys())}"
        if isinstance(v, dict):
            ph = ori_configs[k] if isinstance(ori_configs.get(k), dict) else {}
            ori_configs[k] = update_configs(v, ph)
        else:
            ori_configs[k] = copy.copy(v)
    return ori_configs


def exp_config():
    return {
        "expr_group": "datasetserver_test",
        "expr_name": "oneline_learning_test",
    }


def dataset_config():
    return {"episode_capacity": 100000, "learning_start": 500}


def parameter_server_config():
    return {}


@pytest.mark.parametrize(
    "config_path,mode",
    [
        # ("examples/configs/gym/dqn_cartpole.yaml", "async"),
        ("examples/configs/gym/sac_hopper.yaml", "async"),
    ],
)
class TestOnlineLearning:
    @pytest.fixture
    def env_desc(self):
        env = GymEnv(env_id="Hopper-v2")
        observation_spaces = env.observation_spaces
        action_spaces = env.action_spaces
        possible_agents = env.possible_agents

        def data_shape_func(agent):
            preprosessor = get_preprocessor(observation_spaces[agent])(
                observation_spaces[agent]
            )
            return {
                Episode.CUR_OBS: preprosessor.shape,
                Episode.NEXT_OBS: preprosessor.shape,
                Episode.ACTION_DIST: action_spaces[agent].shape,
                Episode.ACTION: action_spaces[agent].shape,
                Episode.DONE: (),
                Episode.REWARD: (),
            }

        agent_data_shapes = {agent: data_shape_func(agent) for agent in possible_agents}
        env.close()

        return {
            "creator": GymEnv,
            "config": {
                "env_id": "Hopper-v2",
                "scenario_configs": {},
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "data_shapes": agent_data_shapes,
            },
            "possible_agents": possible_agents,
        }

    @pytest.fixture
    def dataset_server(self):
        if not ray.is_initialized():
            ray.init()
        return OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
        ).remote(dataset_config(), exp_config())

    @pytest.fixture
    def parameter_server(self):
        if not ray.is_initialized():
            ray.init()
        return ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(exp_cfg=exp_config(), **parameter_server_config())

    def test_ind_learner_ind_sample(
        self,
        config_path,
        mode,
        env_desc,
        dataset_server: OfflineDataset,
        parameter_server: ParameterServer,
    ):

        with open(os.path.join(settings.BASE_DIR, config_path), "r") as f:
            yaml_config = yaml.load(f)

        if not ray.is_initialized():
            ray.init()
        # return a fake coordinator
        logging.debug("ray has beed init")

        yaml_config["training"]["interface"]["observation_spaces"] = env_desc["config"][
            "observation_spaces"
        ]
        yaml_config["training"]["interface"]["action_spaces"] = env_desc["config"][
            "action_spaces"
        ]

        yaml_config = update_configs(yaml_config)

        exp_cfg = logger.start(
            group=yaml_config.get("group", "experiment"),
            name=yaml_config.get("name", "case") + f"_{time.time()}",
        )

        controller = LightCoordinator.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote()

        start = time.time()
        est_time_limit = 3600
        controller.start.remote(yaml_config, env_desc, exp_cfg)

        while time.time() - start <= est_time_limit:
            pass

        dataset_server.shutdown()
        parameter_server.shutdown()
        controller.shutdown()
        ray.shutdown()
