"""
Environment agent interface is designed for rollout and simulation.
"""

import os
from typing import Tuple
import gym
import ray
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from malib import settings
from malib.algorithm.common.policy import Policy
from malib.algorithm import get_algorithm_space
from malib.utils.episode import EpisodeKey
from malib.utils.typing import (
    DataTransferType,
    Status,
    Dict,
    Any,
    Sequence,
    AgentID,
    PolicyID,
    ParameterDescription,
    BehaviorMode,
    List,
)

import pickle as pkl


def _update_weights(agent_interface: "AgentInterface", pid: PolicyID) -> None:
    """Update weights for agent interface.

    :param AgentInterface agent_interface: An environment agent interface instance.
    :param PolicyID pid: Policy id.
    """
    parameter_server = agent_interface.parameter_server
    parameter_descs = agent_interface.parameter_desc_dict
    parameter_buffer = agent_interface.parameter_buffer

    while parameter_buffer[pid] is None:
        parameter_descs[pid].version = agent_interface.parameter_version[pid]
        task = parameter_server.pull.remote(parameter_descs[pid], keep_return=True)
        status, content = ray.get(task)

        if content.data is not None:
            parameter_buffer[pid] = content.data
            agent_interface.parameter_version[pid] = content.version
            parameter_descs[pid].lock = status.locked
        else:
            break


@dataclass
class AgentInterface:
    """AgentInterface for rollout worker. This interface integrates a group of methods to request parameter from remote
    server, and rollout.
    """

    agent_id: AgentID
    """Environment agent id"""

    observation_space: gym.spaces.Space
    """Raw observation space"""

    action_space: gym.spaces.Space
    """Raw action space"""

    parameter_server: Any
    """Remote parameter server, it is a ray actor"""

    policies: Dict[PolicyID, Policy] = None
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
        self.thread_pool = ThreadPoolExecutor()
        self.parameter_version = defaultdict(lambda: -1)
        self.parameter_buffer = defaultdict(lambda: None)
        self.behavior_mode = BehaviorMode.EXPLORATION

        self._behavior_policy = None

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

        self.sample_dist = dict.fromkeys(self.policies, 1.0 / len(self.policies))
        self.sample_dist.update(distribution)

    def set_behavior_mode(
        self,
        behavior_mode: BehaviorMode = BehaviorMode.EXPLORATION,
    ) -> None:
        """Set behavior mode

        :param BehaviorMode behavior_mode: Behavior mode (BehaviorMode.EXPLORATION, BehaviorMode.EXPLOITABILITY)
        """

        self.behavior_mode = behavior_mode

    def reset(self, policy_id=None, sample_dist=None) -> None:
        """Reset agent interface. Sample dist will reseted with a no-none given `sample_dist`. If `self.sample_dist` is None or `self.policies` has changed length. Then `self.sample_dist` will be reset as uniform."""

        if sample_dist is not None:
            self.sample_dist = sample_dist
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

        :param PolicyID policy_id: Policy id.
        :param Dict[str,Any] policy_description: Policy description, used to create policy instance
        :param ParameterDescription parameter_desc: Parameter description, used to coordinate with parameter server.
        """

        # create policy and pull parameter from remote
        if policy_id in self.policies:
            return
        policy = get_algorithm_space(policy_description["registered_name"]).policy(
            **policy_description,
            env_agent_id=env_aid,
        )
        self.policies[policy_id] = policy
        self.parameter_desc_dict[policy_id] = ParameterDescription(
            time_stamp=parameter_desc.time_stamp,
            identify=parameter_desc.identify,
            env_id=parameter_desc.env_id,
            id=parameter_desc.id,
            type=parameter_desc.type,
            lock=False,
            description=parameter_desc.description.copy(),
            data=None,
            parallel_num=parameter_desc.parallel_num,
            version=-1,
        )

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
    ) -> Tuple[DataTransferType, DataTransferType, List[DataTransferType]]:
        """Return an action by calling `compute_action` of a policy instance.

        :param args: list of args
        :param kwargs: dict of args
        :return: A tuple of action, action_dist, a list of rnn_state
        """
        policy_id = kwargs.get("policy_id", self.behavior_policy)
        if policy_id is None:
            policy_id = self._random_select_policy()
            self._behavior_policy = policy_id
        kwargs.update({"behavior_mode": self.behavior_mode})
        kwargs[EpisodeKey.CUR_OBS] = np.stack(kwargs[EpisodeKey.CUR_OBS])
        kwargs[EpisodeKey.ACTION_MASK] = np.stack(kwargs[EpisodeKey.ACTION_MASK])
        if kwargs.get(EpisodeKey.CUR_STATE) is not None:
            kwargs[EpisodeKey.CUR_STATE] = np.stack(kwargs[EpisodeKey.CUR_STATE])
        rnn_states_list = kwargs[EpisodeKey.RNN_STATE]
        rnn_states_list = list(zip(*rnn_states_list))
        kwargs[EpisodeKey.RNN_STATE] = [
            np.stack(_v) for _v in rnn_states_list if len(_v) > 0
        ]
        rets = self.policies[policy_id].compute_action(*args, **kwargs)
        return rets

    def get_policy(self, pid: PolicyID) -> Policy:
        return self.policies[pid]

    def get_initial_state(
        self, pid=None, batch_size: int = None
    ) -> List[DataTransferType]:
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
        if waiting:
            for pid in pids:
                # wait until task is success or locked
                _update_weights(self, pid)
        else:
            for pid in pids:
                self.thread_pool.submit(_update_weights, self, pid)

        for pid in pids:
            if self.parameter_buffer[pid] is not None:
                self.policies[pid].set_weights(self.parameter_buffer[pid])
                self.parameter_buffer[pid] = None
            status[pid] = (
                Status.LOCKED if self.parameter_desc_dict[pid].lock else Status.SUCCESS
            )
        return status

    def transform_observation(
        self, observation: Any, state: Any, policy_id: PolicyID = None
    ) -> Dict[str, Any]:
        """Transform environment observation with behavior policy's preprocessor. The preprocessed observation will be
        transferred to policy's `compute_action` as an input.

        :param Any observation: Raw environment observation
        :param PolicyID pid: Policy id. Default by None, if specified, agent will switch to policy tagged with `pid`.
        :return: A preprocessed observation.
        """

        policy_id = policy_id or self._random_select_policy()
        policy = self.policies[policy_id]
        if policy.preprocessor is not None:
            return {"obs": policy.preprocessor.transform(observation), "state": state}
        else:
            return {"obs": observation, "state": state}

    def save(self, model_dir: str) -> None:
        """Save policies.

        :param str model_dir: Directory path.
        :return: None
        """

        os.makedirs(model_dir)

        for pid, policy in self.policies.items():
            fp = os.path.join(model_dir, pid + ".pkl")
            with open(fp, "wb") as f:
                pkl.dump(policy, f, protocol=settings.PICKLE_PROTOCOL_VER)

    def close(self):
        """Terminate and do resource recycling"""

        self.thread_pool.shutdown()
