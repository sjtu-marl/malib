import ray
import time
import gym

from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

from malib.utils.logger import Logger
from malib.utils.typing import (
    AgentID,
    Any,
    DataFrame,
    List,
    PolicyID,
    Dict,
    Status,
    BehaviorMode,
)
from malib.algorithm.common.policy import Policy
from malib.envs.agent_interface import _update_weights


@ray.remote
class InferenceWorkerSet:
    def __init__(
        self,
        agent_id: AgentID,
        observation_space: gym.Space,
        action_space: gym.Space,
        parameter_server: Any,
    ) -> None:
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.parameter_sever = parameter_server

        self.policies = {}
        self.sample_dist = {}
        self.parameter_desc_dict = {}
        self.thread_pool = ThreadPoolExecutor()
        self.parameter_version = defaultdict(lambda: -1)
        self.parameter_buffer = defaultdict(lambda: None)
        self.behavior_mode = BehaviorMode.EXPLORATION

        self.request_deque = deque()
        self.response_deque = deque()
        self.client = None
        self.status = Status.IDLE

    def terminate(self):
        self.status = Status.TERMINATE
        self.request_deque.clear()
        self.response_deque.clear()
        self.client = None
        self.thread_pool.shutdown(wait=True)

    def reset_comm(self, client):
        """Reset request deque and response deque"""

        self.request_deque.clear()
        self.response_deque.clear()

        self.client = client
        self.status = Status.IN_PROGRESS
        self.thread_pool.submit(self._compute_action, self)

    def _compute_action(self):
        while self.status == Status.IN_PROGRESS:
            if len(self.request_deque) == 0:
                time.sleep(1)
                continue

            data_frame: DataFrame = self.deque.popleft()

            if data_frame.header.policy_id not in self.policies:
                Logger.warning(
                    "No policy instance with id={} can be found in InferenceWorkerSet={}".format(
                        self.agent_id
                    )
                )
                time.sleep(1)
                continue

            policy: Policy = self.policies[data_frame.header.policy_id]
            rets = policy.compute_action(
                observation=data_frame.data, **data_frame.runtime_config
            )

            self.response_deque.append(
                DataFrame(header=None, data=rets, runtime_config=None)
            )

    def update_weights(
        self, pids: List[PolicyID] = None, waiting: bool = False
    ) -> Dict[PolicyID, Status]:
        """Update weights in async mode.

        :param pids: A list of policy ids, defaults to None
        :type pids: List[PolicyID], optional
        :param waiting: Indicates waiting result or not, defaults to False
        :type waiting: bool, optional
        :return: A dict of policy status, mapping from policy id to status.
        :rtype: Dict[PolicyID, Status]
        """

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

    def request(self, data_frame: DataFrame):
        """Request data frame parsing.

        :param data_frame: A data frame instance
        :type data_frame: DataFrame
        """

        self.deque.append(data_frame)

    def response(self):
        """Send data frame"""


@ray.remote
class RolloutWorkerSet:
    def __init__(self) -> None:
        pass
