import pytest
import ray
import time
import numpy as np

from gym import spaces
from pytest_mock import MockerFixture

from malib import settings

from malib.utils import logger
from malib.utils.typing import (
    List,
    Dict,
    Status,
    TaskDescription,
    TaskRequest,
    TaskType,
    TrainingDescription,
    AgentInvolveInfo,
    TrainingFeedback,
)
from malib.manager.training_manager import TrainingManager
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.agent.agent_interface import AgentFeedback, AgentTaggedFeedback

from tests.parameter_server import FakeParameterServer
from tests.dataset import FakeDataServer
from tests.coordinator import FakeCoordinator


@pytest.fixture(scope="module")
def server_and_config():
    if not ray.is_initialized():
        ray.init()

    # start logger
    exp_cfg = logger.start(
        group="test",
        name=f"test_{time.time()}",
    )
    try:
        coordinator_server = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)
    except ValueError:
        coordinator_server = FakeCoordinator.options(
            name=settings.COORDINATOR_SERVER_ACTOR
        ).remote()

    try:
        parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
    except ValueError:
        parameter_server = FakeParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR
        ).remote()

    try:
        dataset_server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
    except ValueError:
        dataset_server = FakeDataServer.options(
            name=settings.OFFLINE_DATASET_ACTOR
        ).remote()

    agents = [f"agent_{i}" for i in range(4)]
    observation_spaces = {
        agent: spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in agents
    }
    action_spaces = {agent: spaces.Discrete(4) for agent in agents}
    env_desc = {
        "possible_agents": agents,
        "observation_spaces": observation_spaces,
        "action_spaces": action_spaces,
        "config": {"env_id": "test"},
        "creator": None,
    }

    return {
        "coordinator": coordinator_server,
        "parameter_server": parameter_server,
        "dataset_server": dataset_server,
        "exp_cfg": exp_cfg,
        "env_desc": env_desc,
    }


@ray.remote
def gen_fake_remote_task(value=None, p=None):
    time.sleep(0.1)
    if p is not None:
        assert isinstance(value, list)
        idx = np.random.choice(len(value), p=p)
        print("idx and selected value:", idx, value[idx])
        return value[idx]
    else:
        return value


def test_training_manager(server_and_config, mocker: MockerFixture):
    mocker.patch(
        "malib.manager.training_manager.measure_exploitability",
        return_value=(None, None),
    )

    training_agent_mapping = lambda agent: agent
    env_desc = server_and_config["env_desc"]

    manager: TrainingManager = TrainingManager(
        algorithms=None,
        env_desc=env_desc,
        interface_config={
            "type": "independent",
            "observation_spaces": server_and_config["env_desc"]["observation_spaces"],
            "action_spaces": server_and_config["env_desc"]["action_spaces"],
            "use_init_policy_pool": False,
            "population_size": -1,
            "algorithm_mapping": lambda agent: agent,
        },
        training_agent_mapping=training_agent_mapping,
        training_config={},
        exp_cfg=server_and_config["exp_cfg"],
    )

    # suppose all agents share the same policy pool
    policy_tups = [(f"policy_{i}", None) for i in range(3)]

    # mock agent interface
    for agent_id, v in manager._agents.items():
        v.add_policy.remote = mocker.patch.object(
            v.add_policy, "remote", return_value=gen_fake_remote_task.remote()
        )
        v.start.remote = mocker.patch.object(
            v.start, "remote", return_value=gen_fake_remote_task.remote()
        )
        v.get_stationary_state.remote = mocker.patch.object(
            v.get_stationary_state,
            "remote",
            return_value=gen_fake_remote_task.remote(
                AgentTaggedFeedback(id=agent_id, content={agent_id: policy_tups})
            ),
        )
        v.get_policies.remote = mocker.patch.object(
            v.get_policies,
            "remote",
            return_value=gen_fake_remote_task.remote({agent_id: dict(policy_tups)}),
        )
        v.train.remote = mocker.patch.object(v.train, "remote", return_value=None)
        v.require_parameter_desc.remote = mocker.patch.object(
            v.require_parameter_desc,
            "remote",
            return_value=gen_fake_remote_task.remote({agent_id: {}}),
        )

    # ============ test api =================
    manager.init(state_id=str(time.time()))
    assert manager.get_agent_interface_num() == len(manager._agents)

    # check property
    groups: Dict[str, List] = manager.groups

    # add policy to different agent inteface
    interface_ids = list(manager._agents.keys())
    for aid in manager._agents:
        manager.add_policy(
            interface_id=aid,
            task=TaskDescription(
                task_type=TaskType.ADD_POLICY,
                content=TrainingDescription(
                    agent_involve_info=None,
                ),
                state_id=str(time.time()),
            ),
        )

    manager.optimize(
        TaskDescription(
            task_type=TaskType.OPTIMIZE,
            content=TrainingDescription(
                agent_involve_info=AgentInvolveInfo(
                    training_handler=interface_ids[0],
                    trainable_pairs=None,
                    populations=None,
                ),
            ),
            state_id=str(time.time()),
        )
    )

    # two cases: content=AgentFeedback, content=TrainingFeedback
    content = [
        AgentFeedback(id=interface_ids[0], trainable_pairs=None, statistics=None),
        TrainingFeedback(agent_involve_info=None, statistics=None),
    ]
    for e in content:
        manager.retrieve_information(
            TaskRequest(
                task_type=TaskType.OPTIMIZE, content=e, state_id=str(time.time())
            )
        )
    manager.get_exp(None)
    manager.terminate()


def test_rollout_manager(server_and_config, mocker: MockerFixture):
    manager = RolloutWorkerManager(
        rollout_config={
            "num_episodes": 1,
            "num_env_per_worker": 1,
            "worker_num": 2,
            "metric_type": "simple",
        },
        env_desc=server_and_config["env_desc"],
        exp_cfg=server_and_config["exp_cfg"],
    )

    # ======== mocker for rollout worker manager ============
    for worker_idx, worker in manager._workers.items():
        worker.get_status.remote = mocker.patch.object(
            worker.get_status,
            "remote",
            return_value=gen_fake_remote_task.remote(Status.IDLE),
        )
        worker.set_status.remote = mocker.patch.object(
            worker.set_status,
            "remote",
            return_value=gen_fake_remote_task.remote(Status.SUCCESS),
        )
        worker.simulation.remote = mocker.patch.object(
            worker.simulation, "remote", return_value=None
        )
        worker.rollout.remote = mocker.patch.object(
            worker.rollout, "remote", return_value=None
        )
        worker.close.remote = mocker.patch.object(worker.close, "remote")

    # ============ test information retrieve ================
    v = TaskRequest(task_type=TaskType.ROLLOUT, content=None, state_id=str(time.time()))
    ret = manager.retrieve_information(v)
    assert ret == v, "rollout manager should not modify a task request as input"
    print("* task retrieving test passed")

    # ============ test simulation task generation ==========
    manager.simulate(
        TaskDescription(
            task_type=TaskType.SIMULATION, content=None, state_id=str(time.time())
        )
    )
    print("* simulation task generation test passed")

    # ============= test rollout task generation ============
    manager.rollout(
        TaskDescription(
            task_type=TaskType.ROLLOUT, content=None, state_id=str(time.time())
        )
    )
    print("* rollout task generation task passed")
    manager.terminate()


def teardown_module(module):
    ray.shutdown()

    if logger.logger_server is not None:
        logger.terminate()
        logger.logger_server = None
