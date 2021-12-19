import pytest
import ray
import time

from malib import settings
from malib.utils import logger
from malib.utils.general import update_configs
from malib.envs.poker import PokerParallelEnv, env_desc_gen
from malib.backend.coordinator.task import CoordinatorServer
from malib.backend.datapool.offline_dataset_server import OfflineDataset
from malib.backend.datapool.parameter_server import ParameterServer


def setup_module(module):
    if not ray.is_initialized():
        ray.init()


@pytest.fixture(scope="module")
def poker_desc():
    env_description = {
        "creator": PokerParallelEnv,
        "config": {
            "scenario_configs": {"fixed_player": True},
            "env_id": "leduc_poker",
        },
    }

    env = PokerParallelEnv(**env_description["config"])
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    env.close()

    env_description["possible_agents"] = possible_agents
    env_description["observation_spaces"] = observation_spaces
    env_description["action_spaces"] = action_spaces

    return env_description


@pytest.fixture(scope="module")
def algorithms():
    return {
        "DQN": {
            "name": "DQN",
            "custom_config": {
                "gamma": 1.0,
                "eps_min": 0,
                "eps_max": 1.0,
                "eps_anneal_time": 100,
                "lr": 1e-2,
            },
        }
    }


@pytest.fixture(scope="module")
def rollout_config():
    return {
        "type": "async",
        "stopper": "simple_rollout",
        "stopper_config": {"max_step": 10},
        "metric_type": "simple",
        "fragment_length": 100,
        "num_episodes": 1,
        "num_env_per_worker": 1,
        "max_step": 10,
        "postprocessor_types": ["copy_next_frame"],
    }


@pytest.fixture(scope="module")
def evaluation_config():
    return {"max_episode_length": 100, "num_episode": 10}


@pytest.fixture(scope="module")
def global_evaluator_config():
    return {
        "name": "psro",
        "config": {
            "stop_metrics": {"max_iteration": 2, "loss_threshold": 2.0},
        },
    }


@pytest.fixture(scope="module")
def dataset_config():
    return {"episode_capacity": 20000}


def test_leduc_case(
    poker_desc,
    algorithms,
    rollout_config,
    evaluation_config,
    global_evaluator_config,
    dataset_config,
):
    # pack config
    config = dict(
        group="test",
        name="leduc",
        env_description=poker_desc,
        training={
            "interface": {
                "type": "independent",
                "observation_spaces": poker_desc["observation_spaces"],
                "action_spaces": poker_desc["action_spaces"],
            },
            "config": {
                "batch_size": 64,
                "update_interval": 8,
            },
        },
        algorithms=algorithms,
        rollout=rollout_config,
        evaluation=evaluation_config,
        global_evaluator=global_evaluator_config,
        dataset_config=dataset_config,
        task_mode="gt",
    )

    # runner
    config = update_configs(config)

    exp_cfg = logger.start(
        group=config.get("group", "experiment"),
        name=config.get("name", "case") + f"_{time.time()}",
    )

    coordinator_cls = CoordinatorServer.as_remote()

    try:
        offline_dataset = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
    except ValueError:
        offline_dataset = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=1000
        ).remote(config["dataset_config"], exp_cfg)

    try:
        parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
    except ValueError:
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(exp_cfg=exp_cfg, **config["parameter_server"])

    try:
        coordinator_server = ray.get(settings.COORDINATOR_SERVER_ACTOR)
    except ValueError:
        coordinator_server = coordinator_cls.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote(exp_cfg=exp_cfg, **config)

    _ = ray.get(coordinator_server.start.remote())

    while True:
        terminate = ray.get(coordinator_server.is_terminate.remote())
        if terminate:
            break
        else:
            time.sleep(1)


def teardown_module(module):
    logger.terminate()
    ray.shutdown()
    logger.logger_server = None
