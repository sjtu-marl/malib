import pytest
import ray
import time
import shutil
import numpy as np

from gym import spaces

from malib import settings
from malib.envs.agent_interface import AgentInterface
from malib.algorithm import random
from malib.utils.typing import BehaviorMode, ParameterDescription, Status

from tests.parameter_server import FakeParameterServer


@pytest.mark.parametrize(
    "observation_space,action_space",
    [
        (spaces.Box(low=-1.0, high=1.0, shape=(1,)), spaces.Discrete(3)),
        (
            spaces.Box(low=-1.0, high=1.0, shape=(4,)),
            spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        ),
    ],
    scope="class",
)
class TestEnvAgentInterface:
    @pytest.fixture(autouse=True)
    def init(self, observation_space, action_space):
        if not ray.is_initialized():
            ray.init()

        try:
            parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        except ValueError:
            parameter_server = FakeParameterServer.remote()

        self.interface = AgentInterface(
            agent_id="test",
            observation_space=observation_space,
            action_space=action_space,
            parameter_server=parameter_server,
        )

        self.env_id = "test"
        desc = dict(
            registered_name=random.NAME,
            observation_space=observation_space,
            action_space=action_space,
            model_config={},
            custom_config={},
        )
        self.interface.add_policy(
            self.env_id,
            "policy_0",
            desc,
            ParameterDescription(
                time_stamp=time.time(),
                identify="policy_0",
                env_id=self.env_id,
                id="policy_0",
                version=0,
                description=desc,
            ),
        )

    def test_set_behavior_mode(self):
        self.interface.set_behavior_mode(BehaviorMode.EXPLOITATION)
        assert self.interface.behavior_mode == BehaviorMode.EXPLOITATION
        self.interface.set_behavior_mode(BehaviorMode.EXPLORATION)
        assert self.interface.behavior_mode == BehaviorMode.EXPLORATION

    def test_set_behavior_dist(self, observation_space, action_space):
        PPOOL_SIZE = 10
        for i in range(PPOOL_SIZE):
            pid = f"policy_{i}"
            desc = dict(
                registered_name=random.NAME,
                observation_space=observation_space,
                action_space=action_space,
                model_config={},
                custom_config={},
            )
            parameter_desc = ParameterDescription(
                time_stamp=time.time(),
                identify=pid,
                env_id=self.env_id,
                id=pid,
                version=0,
                description=desc,
            )
            self.interface.add_policy(self.env_id, pid, desc, parameter_desc)

        dist = dict(zip([f"policy_{i}" for i in range(PPOOL_SIZE)], [1 / PPOOL_SIZE]))
        self.interface.set_behavior_dist(dist)

    def test_serialize(self):
        # since pickle does not support __post_init__
        # while install dill has no need, we just test __setstate__ and __getstate__
        state = self.interface.__getstate__()
        self.interface.__setstate__(state)

    def test_reset(self, observation_space, action_space):
        PPOOL_SIZE = 10
        for i in range(PPOOL_SIZE):
            pid = f"policy_{i}"
            desc = dict(
                registered_name=random.NAME,
                observation_space=observation_space,
                action_space=action_space,
                model_config={},
                custom_config={},
            )
            parameter_desc = ParameterDescription(
                time_stamp=time.time(),
                identify=pid,
                env_id=self.env_id,
                id=pid,
                version=0,
                description=desc,
            )
            self.interface.add_policy(self.env_id, pid, desc, parameter_desc)

        # reset without any policy figured out?
        self.interface.reset()
        assert self.interface.behavior_policy is not None, (
            self.interface.behavior_policy,
            list(self.interface.policies.keys()),
        )

        # reset with policy figure out?
        self.interface.reset(policy_id="policy_2")
        assert (
            self.interface.behavior_policy == "policy_2"
        ), self.interface.behavior_policy
        assert np.isclose(sum(self.interface.sample_dist.values()), 1.0)
        for e in self.interface.sample_dist.values():
            assert e == 1 / PPOOL_SIZE

        # reset with sample dist figureout?
        sample_dist = [1.0 / PPOOL_SIZE] * PPOOL_SIZE
        sample_dist = dict(zip([f"policy_{i}" for i in range(PPOOL_SIZE)], sample_dist))
        self.interface.reset(sample_dist=sample_dist)
        assert self.interface.behavior_policy is not None

        sample_dist = [0.0] * PPOOL_SIZE
        sample_dist[-1] = 1.0
        sample_dist = dict(zip([f"policy_{i}" for i in range(PPOOL_SIZE)], sample_dist))
        self.interface.reset(sample_dist=sample_dist)
        assert self.interface.behavior_policy == "policy_{}".format(PPOOL_SIZE - 1), (
            self.interface.behavior_policy,
            list(self.interface.policies.keys()),
        )

    def test_compute_action(self, observation_space: spaces.Space):
        self.interface.reset()
        x = self.interface.compute_action(
            observation=np.asarray(
                [observation_space.sample() for _ in range(4)]
            ).reshape((-1,) + observation_space.shape),
            rnn_state=[self.interface.get_initial_state() for _ in range(4)],
        )
        assert len(x) == 3
        action = []
        action_dist = []
        rnn_state = []
        for e in x[0]:
            action.append(e)
        for e in x[1]:
            action_dist.append(e)
        for e in x[2]:
            rnn_state.append(e)
        assert len(action) == len(action_dist) == 4

    def test_weight_update(self):
        self.interface.reset()
        pid = self.interface.behavior_policy
        status = self.interface.update_weights([pid], waiting=True)
        assert isinstance(status, dict) and status[pid] == Status.SUCCESS, status
        status = self.interface.update_weights([pid], waiting=False)
        assert isinstance(status, dict) and status[pid] == Status.SUCCESS, status

    def test_observation_transformation(self, observation_space):
        self.interface.reset()
        res1 = self.interface.transform_observation(observation_space.sample())
        res2 = self.interface.transform_observation(
            [observation_space.sample() for _ in range(4)]
        )
        target_shape = observation_space.shape
        if target_shape == (1,):
            target_shape = ()
        assert res1["obs"].shape == target_shape
        assert res2["obs"].shape == (4,) + target_shape

    def test_save_and_close(self):
        self.interface.reset()
        pid = self.interface.behavior_policy
        tmp_dir = "/tmp/model_{}".format(time.time())
        self.interface.save(tmp_dir)
        self.interface.close()
        shutil.rmtree(tmp_dir)

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
