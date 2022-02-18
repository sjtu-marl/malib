import ray
import gym
import numpy as np

from dataclasses import dataclass

from malib import settings
from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    PolicyID,
    BehaviorMode,
    ParameterDescription,
    Status,
    Sequence,
)
from malib.utils.logger import Log
from malib.utils.episode import EpisodeKey
from malib.utils.preprocessor import get_preprocessor

from malib.algorithm.random.policy import RandomPolicy


class ServerMixin:
    def init_coordinator(self):
        server = None

        try:
            server = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)
        except ValueError:
            from tests.coordinator import FakeCoordinator

            server = FakeCoordinator.options(
                name=settings.COORDINATOR_SERVER_ACTOR
            ).remote()

        return server

    def init_dataserver(self):
        server = None

        try:
            server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
        except ValueError:
            from tests.dataset import FakeDataServer

            server = FakeDataServer.options(
                name=settings.OFFLINE_DATASET_ACTOR
            ).remote()

        return server

    def init_parameter_server(self):
        server = None
        try:
            server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        except ValueError:
            from tests.parameter_server import FakeParameterServer

            server = FakeParameterServer.options(
                name=settings.PARAMETER_SERVER_ACTOR
            ).remote()
        return server


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
            self.policies = {
                "policy_0": RandomPolicy(
                    "test", self.observation_space, self.action_space, {}, {}
                )
            }

        # sample distribution over policy set
        self.sample_dist = {}
        # parameter description dictionary
        self.parameter_desc_dict = {
            "policy_0": ParameterDescription.gen_template(id="policy_0")
        }

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


class FakeStepping:
    def __init__(
        self,
        exp_cfg: Dict[str, Any],
        env_desc: Dict[str, Any],
        dataset_server=None,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types=["default"],
    ):
        pass

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization."""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces,
        fragment_length,
        desc,
        buffer_desc=None,
    ):
        task_type = desc["flag"]
        rollout_results = {"total_fragment_length": 100, "eval_info": {}}
        return task_type, rollout_results
