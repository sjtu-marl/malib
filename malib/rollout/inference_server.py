from functools import reduce
from operator import mul
import os
import pickle as pkl
import time
import threading
import gc

from collections import deque, defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
import traceback
from types import LambdaType

import ray
import gym
import numpy as np

from ray.util.queue import Queue

from malib import settings
from malib.utils.logger import Logger
from malib.utils.typing import (
    AgentID,
    Any,
    DataFrame,
    List,
    MetaParameterDescription,
    ParameterDescription,
    PolicyID,
    Dict,
    Status,
    BehaviorMode,
    Union,
)
from malib.algorithm import get_algorithm_space
from malib.algorithm.common.policy import Policy
from malib.utils.episode import EpisodeKey


RuntimeHandler = namedtuple("RuntimeHandler", "sender,recver,runtime_config,rnn_states")


@ray.remote
class InferenceWorkerSet:
    def __init__(
        self,
        agent_id: AgentID,
        observation_space: gym.Space,
        action_space: gym.Space,
        parameter_server: Any,
        governed_agents: List[AgentID],
        force_weight_update: bool = False,
    ) -> None:
        self.runtime_agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.parameter_server = parameter_server

        self.parameter_desc: List[ParameterDescription] = []
        self.parameter_version: List[int] = []

        self.thread_pool = ThreadPoolExecutor()
        self.parameter_buffer_lock = threading.Lock()
        self.parameter_buffer = defaultdict(lambda: None)
        self.governed_agents = governed_agents
        self.policies = {aid: {} for aid in governed_agents}

        self.runtime: Dict[int, RuntimeHandler] = {}

        Logger.info("ready to submit weights update")
        self.thread_pool.submit(_update_weights, self, force_weight_update)

    def shutdown(self):
        self.thread_pool.shutdown(wait=True)
        for _runtime in self.runtime.values():
            _runtime.sender.shutdown(True)
            _runtime.recver.shutdown(True)
        self.runtime: Dict[int, RuntimeHandler] = {}

    def save(self, model_dir: str) -> None:
        """Save policies.

        :param str model_dir: Directory path.
        :return: None
        """

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for pid, policy in self.policies.items():
            fp = os.path.join(model_dir, pid + ".pkl")
            with open(fp, "wb") as f:
                pkl.dump(policy, f, protocol=settings.PICKLE_PROTOCOL_VER)

    def connect(
        self,
        queues: List[Queue],
        runtime_config: Dict[str, Any],
        runtime_id: int,
    ):
        """Connect new inference task with given configuraiton and queus.

        :param queues: A tuple of send and recieve queus.
        :type queues: List[Queue]
        :param runtime_config: The runtime configuration, including parameter description dict.
        :type runtime_config: Dict[str, Any]
        :param runtime_id: The referred runtime id.
        :type runtime_id: int
        """

        send_queue, recv_queue = queues
        is_exisiting = runtime_id in self.runtime
        self.runtime[runtime_id] = RuntimeHandler(
            send_queue,
            recv_queue,
            runtime_config,
            {agent: [] for agent in self.governed_agents},
        )

        # print("agent server: {} accepts a connection with: {}".format(self.runtime_agent_id, runtime_id))

        with self.parameter_buffer_lock:
            # should be a mapped parameter description dict
            parameter_desc_dict: Dict[
                AgentID, Union[ParameterDescription, MetaParameterDescription]
            ] = runtime_config["parameter_desc_dict"][self.runtime_agent_id]
            # for aid, p_desc in parameter_desc_dict.items():
            for aid, p_desc in parameter_desc_dict.items():
                if isinstance(p_desc, ParameterDescription):
                    self._update_policies(p_desc, aid)
                elif isinstance(p_desc, MetaParameterDescription):
                    for parameter_desc in p_desc.parameter_desc_dict.values():
                        self._update_policies(parameter_desc, aid)

        if not is_exisiting:
            self.thread_pool.submit(_compute_action, self, runtime_id)

    def _update_policies(self, p_desc: ParameterDescription, aid: AgentID):
        if p_desc.id not in self.policies[aid]:
            policy = get_algorithm_space(p_desc.description["registered_name"]).policy(
                **p_desc.description, env_agent_id=aid
            )
            self.policies[aid][p_desc.id] = policy
            self.parameter_desc.append(
                ParameterDescription(
                    time_stamp=p_desc.time_stamp,
                    identify=p_desc.identify,
                    env_id=p_desc.env_id,
                    id=p_desc.id,
                    type=p_desc.type,
                    lock=False,
                    description=p_desc.description.copy(),
                    data=None,
                    parallel_num=p_desc.parallel_num,
                    version=-1,
                )
            )
            self.parameter_version.append(-1)


def _sample_policy_id(runtime_config, main_agent_id, old_policy_id):
    policy_id = runtime_config["main_behavior_policies"].get(main_agent_id, None)
    dist = runtime_config["policy_distribution"]
    sample_mode = runtime_config["sample_mode"]

    if policy_id:
        return policy_id

    if sample_mode == "once" and old_policy_id is not None:
        policy_id = old_policy_id
    else:
        pids = list(dist[main_agent_id].keys())
        probs = list(dist[main_agent_id].values())
        policy_id = np.random.choice(pids, p=probs)

    return policy_id


def _get_initial_states(self, runtime_id, observation, policy: Policy, identifier):
    # try to retrive cached states
    if len(self.runtime[runtime_id].rnn_states[identifier]) > 0:
        return self.runtime[runtime_id].rnn_states[identifier][-1]
    else:
        # use inner shape to judge it
        offset = len(policy.preprocessor.shape)
        if offset < len(observation.shape):
            batch_size = reduce(mul, observation.shape[:-offset])
        else:
            batch_size = 1
        return policy.get_initial_state(batch_size=batch_size)


def _update_initial_states(self, runtime_id, rnn_states, identifier):
    self.runtime[runtime_id].rnn_states[identifier].append(rnn_states)


def _compute_action(self: InferenceWorkerSet, runtime_id: int):

    try:
        handler = self.runtime[runtime_id]
        runtime_config = handler.runtime_config
        policy_id = None

        # print("start compute action thread for runtime={} agent={}".format(runtime_id, self.runtime_agent_id))
        while True:

            if handler.recver.empty():
                continue

            data_frames: List[DataFrame] = handler.recver.get()
            rets = {}
            send_data_frames = []
            with self.parameter_buffer_lock:
                for data_frame in data_frames:
                    policy_id = _sample_policy_id(
                        runtime_config, data_frame.identifier, policy_id
                    )
                    policy: Policy = self.policies[data_frame.identifier][policy_id]
                    kwargs = {**data_frame.data, **data_frame.runtime_config}
                    batch_size = len(kwargs["environment_ids"])
                    assert EpisodeKey.CUR_OBS in kwargs, kwargs.keys()
                    observation = kwargs.pop(EpisodeKey.CUR_OBS)
                    # if EpisodeKey.RNN_STATE not in kwargs:
                    kwargs[EpisodeKey.RNN_STATE] = _get_initial_states(
                        self,
                        runtime_id,
                        observation,
                        policy,
                        identifier=data_frame.identifier,
                    )
                    (
                        rets[EpisodeKey.ACTION],
                        rets[EpisodeKey.ACTION_DIST],
                        rets[EpisodeKey.RNN_STATE],
                    ) = policy.compute_action(observation=observation, **kwargs)
                    # compute state value
                    rets[EpisodeKey.STATE_VALUE] = policy.value_function(
                        observation=observation,
                        action_dist=rets[EpisodeKey.ACTION_DIST].copy(),
                        **kwargs
                    )
                    for k, v in rets.items():
                        if k == EpisodeKey.RNN_STATE:
                            continue
                        if len(v.shape) < 1:
                            rets[k] = v.reshape(-1)
                        elif v.shape[0] == 1:
                            continue
                        else:
                            rets[k] = v.reshape(batch_size, -1)
                    # recover env length
                    send_df = DataFrame(
                        identifier=data_frame.identifier,
                        data=rets,
                        runtime_config=data_frame.runtime_config,
                    )
                    send_data_frames.append(send_df)
                    # save rnn state
                    _update_initial_states(
                        self,
                        runtime_id,
                        rets[EpisodeKey.RNN_STATE],
                        identifier=data_frame.identifier,
                    )
                    # print("done for compute action")

            handler.sender.put_nowait(send_data_frames)
    except Exception as e:
        # print(data_frame.data.keys())
        traceback.print_exc()
        raise e


def _update_weights(self: InferenceWorkerSet, force: bool = False) -> None:
    """Update weights for agent interface."""
    try:
        parameter_server = self.parameter_server
        parameter_descs = self.parameter_desc
        # parameter_buffer = self.parameter_buffer
        parameter_version = self.parameter_version

        while True:
            with self.parameter_buffer_lock:
                tasks = []
                for version, p_desc in zip(parameter_version, parameter_descs):
                    # set request version
                    if force:
                        p_desc.version = -1
                    else:
                        p_desc.version = version
                    task = parameter_server.pull.remote(p_desc, keep_return=True)
                    tasks.append(task)
                rets = ray.get(tasks)
                for i, (status, content) in enumerate(rets):
                    if content.data is not None:
                        # update parameter here
                        print(
                            "update weights here with version:",
                            content.version,
                            content.identify,
                        )
                        self.policies[content.identify][
                            parameter_descs[i].id
                        ].set_weights(content.data)
                        parameter_version[i] = content.version
                        parameter_descs[i].lock = status.locked
            gc.collect()
            time.sleep(1)
    except Exception as e:
        traceback.print_exc()
