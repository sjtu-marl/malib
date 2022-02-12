import time
import pytest
import gym
import numpy as np

from pytest_mock import MockerFixture
from pytest import MonkeyPatch
from dataclasses import dataclass

from malib.utils.typing import (
    AgentInvolveInfo,
    Dict,
    Any,
    RolloutDescription,
    TaskDescription,
    TaskType,
    AgentID,
    PolicyID,
    BehaviorMode,
    ParameterDescription,
    Status,
    Sequence,
)
from malib.utils.episode import EpisodeKey
from malib.utils.preprocessor import get_preprocessor

from tests import ServerMixin


def simple_env_desc():
    n_agents = 2
    agent_ids = [f"agent_{i}" for i in range(n_agents)]
    action_spaces = dict(
        zip(agent_ids, [gym.spaces.Discrete(2) for i in range(n_agents)])
    )
    obs_spaces = dict(
        zip(
            agent_ids,
            [gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)) for _ in range(n_agents)],
        )
    )

    return {
        "creator": None,
        "possible_agents": agent_ids,
        "action_spaces": action_spaces,
        "observation_spaces": obs_spaces,
        "config": {"env_id": "test", "scenario_configs": None},
    }


@dataclass
class FakeAgentInterface:
    agent_id: AgentID
    """Environment agent id"""

    observation_space: gym.spaces.Space
    """Raw observation space"""

    action_space: gym.spaces.Space
    """Raw action space"""

    parameter_server: Any
    """Remote parameter server, it is a ray actor"""

    policies: Dict[PolicyID, "Policy"] = None
    """A dict of policies"""

    sample_dist: Dict[PolicyID, float] = None
    """Policy behavior distribution"""

    def __getstate__(self):
        """Return state without heavy initialization.

        :return: a dict
        """

        real_state = {}
        for k, v in self.__dict__.copy().items():
            if k == "parameter_buffer":
                for pid in v:
                    if v[pid] is not None:
                        self.policies[pid].set_weights(v[pid])
                        v[pid] = None
            if k in [
                "thread_pool",
                "task_status",
                "parameter_version",
            ]:
                continue
            # update parameters if possible
            if k == "parameter_buffer":
                for pid in self.policies:
                    if v[pid] is not None:
                        self.policies[pid].set_weights(v[pid])
            real_state[k] = v

        return real_state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __post_init__(self):
        if self.policies is None:
            self.policies = {}

        # sample distribution over policy set
        self.sample_dist = {}
        # parameter description dictionary
        self.parameter_desc_dict = {}

        self._behavior_policy = None
        self.preprocessor = get_preprocessor(self.observation_space)

    @property
    def behavior_policy(self) -> PolicyID:
        return self._behavior_policy

    def set_behavior_dist(self, distribution: Dict[PolicyID, float]) -> None:
        """Set policy distribution with given distribution. The distribution has no need to cover all policies in this
        agent.

        :param distribution: Dict[PolicyID, float], a dict describes the policy distribution, the summation should be 1
        """

        for pid in distribution:
            assert pid in self.policies, (pid, list(self.policies.keys()))

        self.sample_dist = {}  # dict.fromkeys(self.policies, 1.0 / len(self.policies))
        self.sample_dist.update(distribution)

    def set_behavior_mode(
        self,
        behavior_mode: BehaviorMode = BehaviorMode.EXPLORATION,
    ) -> None:
        """Set behavior mode

        :param BehaviorMode behavior_mode: Behavior mode (BehaviorMode.EXPLORATION, BehaviorMode.EXPLOITABILITY)
        """

        self.behavior_mode = behavior_mode

    def reset(self, policy_id=None, sample_dist: Dict[PolicyID, float] = None) -> None:
        """Reset agent interface.

        Sample dist will be reset with a no-none given `sample_dist`.
        If `self.sample_dist` is None or `self.policies` has changed length.
        Then `self.sample_dist` will be reset as an uniform.

        The, the behavior policy will be reset. If the given `policy_id` is not None, the agent
        will select policy identified with `policy_id` as its behavior_policy. Or, random select one
        with its behavior distribution.
        """

        if sample_dist is not None:
            assert isinstance(sample_dist, dict)
            self.sample_dist = sample_dist

        # len(sample_dist) != len(self.policies) means that some new policies were added,
        # and the sample dist has not beeen reset yet.
        if self.sample_dist is None or len(self.sample_dist) != len(self.policies):
            self.sample_dist = dict.fromkeys(self.policies, 1.0 / len(self.policies))

        if policy_id is not None:
            assert policy_id in self.policies
            self._behavior_policy = policy_id
        else:
            self._behavior_policy = self._random_select_policy()

        # then reset sampled behavior_policy
        policy = self.policies[self._behavior_policy]
        # reset intermediate states, e.g. rnn states
        policy.reset()

    def add_policy(
        self,
        env_aid,
        policy_id: PolicyID,
        policy_description: Dict[str, Any],
        parameter_desc: ParameterDescription,
    ) -> None:
        """Add new policy to policy set, if policy id existed, will return a existing policy.

        Note: users must to call `reset` to ensure the consistency between `sample_dist` and `policies` before using.

        :param PolicyID policy_id: Policy id.
        :param Dict[str,Any] policy_description: Policy description, used to create policy instance
        :param ParameterDescription parameter_desc: Parameter description, used to coordinate with parameter server.
        """

        # create policy and pull parameter from remote
        if policy_id in self.policies:
            return
        self.policies[policy_id] = None

    def _random_select_policy(self) -> PolicyID:
        """Random select a policy, and return its id."""
        assert len(self.policies) > 0, "No available policies."
        res = str(
            np.random.choice(
                list(self.sample_dist.keys()), p=list(self.sample_dist.values())
            )
        )
        return res

    def compute_action(
        self, *args, **kwargs
    ):  # Tuple[DataTransferType, DataTransferType, List[DataTransferType]]:
        """Return a batch of action by calling `compute_action` of a policy instance.
        Args contains a batch of data.

        :param args: list of args
        :param kwargs: dict of args
        :return: A tuple of action, action_dist, a list of rnn_state
        """
        policy_id = kwargs.get("policy_id", self.behavior_policy)
        # by default, we do not allow dynamic policy id in an episode.
        assert policy_id is not None
        # if policy_id is None:
        #     policy_id = self._random_select_policy()
        #     self._behavior_policy = policy_id
        kwargs.update({"behavior_mode": self.behavior_mode})
        kwargs[EpisodeKey.CUR_OBS] = np.stack(kwargs[EpisodeKey.CUR_OBS])
        if EpisodeKey.ACTION_MASK in kwargs:
            kwargs[EpisodeKey.ACTION_MASK] = np.stack(kwargs[EpisodeKey.ACTION_MASK])
        if kwargs.get(EpisodeKey.CUR_STATE) is not None:
            kwargs[EpisodeKey.CUR_STATE] = np.stack(kwargs[EpisodeKey.CUR_STATE])
        rnn_states_list = kwargs[EpisodeKey.RNN_STATE]
        rnn_states_list = list(zip(*rnn_states_list))
        kwargs[EpisodeKey.RNN_STATE] = [
            np.stack(_v) for _v in rnn_states_list if len(_v) > 0
        ]
        # XXX(ming): how about EpisodeKey.RNN_MASK ?
        rets = self.policies[policy_id].compute_action(*args, **kwargs)

        # convert rets to iteratable
        rets = (iter(rets[0]), iter(rets[1]), [iter(v) for v in rets[2]])
        return rets

    def get_policy(self, pid: PolicyID):
        return self.policies[pid]

    def get_initial_state(self, pid=None, batch_size: int = None):
        """Return a list of initial rnn states"""

        pid = pid or self.behavior_policy
        assert pid is not None, "Behavior policy or input pid cannot both be None"
        res = self.policies[pid].get_initial_state(batch_size)
        return res

    def update_weights(
        self, pids: Sequence = None, waiting: bool = False
    ) -> Dict[PolicyID, Status]:
        """Update weight in async mode"""

        pids = pids or list(self.policies.keys())
        status = {}

        for pid in pids:
            if self.parameter_buffer[pid] is not None:
                self.policies[pid].set_weights(self.parameter_buffer[pid])
                self.parameter_buffer[pid] = None
            status[pid] = Status.LOCKED
        return status

    def transform_observation(
        self, observation: Any, state: Any = None, policy_id: PolicyID = None
    ) -> Dict[str, Any]:
        return {
            "obs": self.preprocessor.transform(observation).squeeze(),
            "state": state,
        }

    def save(self, model_dir: str) -> None:
        """Save policies.

        :param str model_dir: Directory path.
        :return: None
        """

        pass

    def close(self):
        """Terminate and do resource recycling"""

        pass


@pytest.mark.parametrize(
    "env_desc,kwargs",
    [
        (
            simple_env_desc(),
            {
                "num_rollout_actors": 2,
                "num_eval_actors": 1,
                "exp_cfg": {},
                "use_subproc_env": False,
                "batch_mode": False,
                "postprocessor_types": ["default"],
            },
        )
    ],
)
class TestRolloutWorker(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(
        self,
        env_desc: Dict[str, Any],
        kwargs: Dict,
        mocker: MockerFixture,
        monkeypatch: MonkeyPatch,
    ):
        self.locals = locals()
        self.coordinator = self.init_coordinator()
        self.parameter_server = self.init_parameter_server()
        self.dataset_server = self.init_dataserver()
        # mute remote logger
        monkeypatch.setattr("malib.settings.USE_REMOTE_LOGGER", False)
        monkeypatch.setattr("malib.settings.USE_MONGO_LOGGER", False)
        # XXX(ming): mock AgentInterface directly will raise deep recursive error here.
        monkeypatch.setattr(
            "malib.rollout.base_worker.AgentInterface", FakeAgentInterface
        )
        from malib.rollout.rollout_worker import RolloutWorker

        self.worker = RolloutWorker("test", env_desc, **kwargs)

    def test_actor_pool_checking(self):
        num_eval_actors = self.locals["kwargs"]["num_eval_actors"]
        num_rollout_actors = self.locals["kwargs"]["num_rollout_actors"]

        assert len(self.worker.actors) == num_eval_actors + num_rollout_actors

    def test_simulation_exec(self):
        task_desc = TaskDescription.gen_template(
            task_type=TaskType.SIMULATION,
            state_id="test_{}".format(time.time()),
            content={},
        )
        self.worker.simulation(task_desc)

    def test_rollout_exec(self):
        task_desc = TaskDescription.gen_template(
            task_type=TaskType.ROLLOUT,
            state_id="test_{}".format(time.time()),
            content={},
        )
        self.worker.rollout(task_desc)
