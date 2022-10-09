# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, List, Dict, Tuple
from functools import reduce
from operator import mul
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from readerwriterlock import rwlock

import os
import time
import traceback

import pickle as pkl
import ray
import gym
import numpy as np

from ray.util.queue import Queue

from malib import settings
from malib.remote.interface import RemoteInterface
from malib.utils.typing import AgentID, DataFrame, PolicyID
from malib.utils.logging import Logger
from malib.utils.timing import Timing
from malib.utils.episode import Episode
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.common.policy import Policy
from malib.backend.parameter_server import ParameterServer


ClientHandler = namedtuple("ClientHandler", "sender,recver,runtime_config,rnn_states")


class InferenceWorkerSet(RemoteInterface):
    def __init__(
        self,
        agent_id: AgentID,
        observation_space: gym.Space,
        action_space: gym.Space,
        parameter_server: ParameterServer,
        governed_agents: List[AgentID],
    ) -> None:
        self.runtime_agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.parameter_server = parameter_server

        self.thread_pool = ThreadPoolExecutor()
        self.governed_agents = governed_agents
        self.policies: Dict[str, Policy] = {}
        self.strategy_spec_dict: Dict[str, StrategySpec] = {}

        self.clients: Dict[int, ClientHandler] = {}
        self.parameter_buffer_lock = Lock()

        marker = rwlock.RWLockFair()

        self.shared_wlock = marker.gen_wlock()
        self.thread_pool.submit(_update_weights, self)
        self.thread_pool.submit(_compute_action, self, False, marker.gen_rlock())
        self.thread_pool.submit(_compute_action, self, True, marker.gen_rlock())

    def shutdown(self):
        self.thread_pool.shutdown(wait=True)
        for _handler in self.clients.values():
            _handler.sender.shutdown(True)
            _handler.recver.shutdown(True)
        self.clients: Dict[int, ClientHandler] = {}

    def save(self, model_dir: str) -> None:
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
        client_id: int,
    ):
        """Connect new inference task with given configuration and queues.

        Args:
            queues (List[Queue]): A list of send and recieve queues.
            runtime_config (Dict[str, Any]): Runtime configuration for rollout.
            client_id (int): Refered client id.
        """

        try:
            send_queue, recv_queue = queues
            with self.shared_wlock:
                self.clients[client_id] = ClientHandler(
                    send_queue,
                    recv_queue,
                    runtime_config,
                    {agent: [] for agent in self.governed_agents},
                )

                strategy_spec: StrategySpec = runtime_config["strategy_specs"][
                    self.runtime_agent_id
                ]
                self._update_policies(strategy_spec, self.runtime_agent_id)
        except Exception as e:
            Logger.error(traceback.format_exc())

    def _update_policies(self, strategy_spec: StrategySpec, agent_id: AgentID):
        for strategy_spec_pid in strategy_spec.policy_ids:
            policy_id = f"{strategy_spec.id}/{strategy_spec_pid}"
            if policy_id not in self.policies:
                policy = strategy_spec.gen_policy()
                self.policies[policy_id] = policy


def _get_initial_states(self, client_id, observation, policy: Policy, identifier):
    # FIXME(ming): KeyError: None
    if (
        client_id is not None
        and len(self.clients[client_id].rnn_states[identifier]) > 0
    ):
        return self.clients[client_id].rnn_states[identifier][-1]
    else:
        # use inner shape to judge it
        offset = len(policy.preprocessor.shape)
        if offset < len(observation.shape):
            batch_size = reduce(mul, observation.shape[:-offset])
        else:
            batch_size = 1
        return policy.get_initial_state(batch_size=batch_size)


def _update_initial_states(self, client_id, rnn_states, identifier):
    """Maintain the intermediate states produced by policy, for session each.

    Args:
        client_id (int): Client id.
        rnn_states (Any): A tuple of states
        identifier (str): Identifier, agent id in general.
    """

    self.clients[client_id].rnn_states[identifier].append(rnn_states)


def _compute_action(self: InferenceWorkerSet, eval_mode: bool, reader_lock: Any):
    """Maintain the session of action compute for runtime handler tagged with `runtime_id`.

    Args:
        self (InferenceWorkerSet): The instance of inference server, it is actually a ray.ObjectRef.
        eval_mode (bool): Either current thread running for eval mode or not.

    Raises:
        e: Any expectation.
    """

    timer = Timing()

    try:
        while True:
            policy_refer_data_frames: Dict[str, List[DataFrame]] = defaultdict(
                lambda: []
            )
            client_responses: Dict[str, List[DataFrame]] = {}
            client_agent_seg_tups: List[Tuple[str, AgentID, int]] = []
            dataframe_meta_data_buffers: Dict[str, Dict[AgentID, Dict]] = {}

            with reader_lock:
                with timer.time_avg("ready"):
                    for client_id, handler in self.clients.items():
                        if handler.runtime_config["evaluate"] is not eval_mode:
                            continue
                        if handler.recver.empty():
                            continue
                        dataframe_meta_data_buffers[client_id] = {}
                        dataframes: List[DataFrame] = handler.recver.get_nowait()
                        strategy_specs: Dict[
                            AgentID, StrategySpec
                        ] = handler.runtime_config["strategy_specs"]
                        for dataframe in dataframes:
                            agent_id = dataframe.identifier
                            spec = strategy_specs[agent_id]
                            batch_size = len(dataframe.meta_data["environment_ids"])
                            spec_policy_id = spec.sample()
                            policy_id = f"{spec.id}/{spec_policy_id}"
                            policy_refer_data_frames[policy_id].append(dataframe)
                            client_agent_seg_tups.append(
                                (client_id, agent_id, batch_size)
                            )
                            dataframe_meta_data_buffers[client_id][
                                agent_id
                            ] = dataframe.meta_data.copy()

            if len(policy_refer_data_frames) == 0:
                time.sleep(0.5)
                continue

            merged_dataframes: Dict[PolicyID, DataFrame] = merge_data_frames(
                policy_refer_data_frames,
                preset_meta_data={"evaluate": eval_mode},
            )

            for pid, dataframe in merged_dataframes.items():
                policy: Policy = self.policies[pid]
                kwargs = {
                    Episode.DONE: dataframe.data[Episode.DONE],
                    Episode.ACTION_MASK: dataframe.data[Episode.ACTION_MASK],
                    "evaluate": dataframe.meta_data["evaluate"],
                }
                batch_size = len(dataframe.meta_data["environment_ids"])
                observation = dataframe.data[Episode.CUR_OBS]

                # FIXME(ming): not identifier dependent
                kwargs[Episode.RNN_STATE] = _get_initial_states(
                    self,
                    None,
                    observation,
                    policy,
                    identifier=dataframe.identifier,
                )

                rets = {}
                with timer.time_avg("compute_action"):
                    (
                        rets[Episode.ACTION],
                        rets[Episode.ACTION_LOGITS],
                        rets[Episode.ACTION_DIST],
                        rets[Episode.RNN_STATE],
                    ) = policy.compute_action(observation=observation, **kwargs)

                # compute state value
                with timer.time_avg("compute_value"):
                    rets[Episode.STATE_VALUE] = policy.value_function(
                        observation=observation,
                        action_dist=rets[Episode.ACTION_DIST].copy(),
                        **kwargs,
                    )
                for k, v in rets.items():
                    if k == Episode.RNN_STATE:
                        continue
                    if len(v.shape) < 1:
                        rets[k] = v.reshape(-1)
                    elif v.shape[0] == 1:
                        continue
                    else:
                        rets[k] = v.reshape(batch_size, -1)

                unmerge_policy_rets(
                    client_responses,
                    rets,
                    client_agent_seg_tups,
                    dataframe_meta_data_buffers,
                )

            # recover rets to response handler
            with reader_lock:
                with timer.time_avg("unmerge"):
                    for client_id, agent_dataframes in client_responses.items():
                        self.clients[client_id].sender.put_nowait(agent_dataframes)
            # print(f"timer information: {timer.todict()}")
    except Exception as e:
        traceback.print_exc()
        raise e


def _update_weights(inference_server: InferenceWorkerSet) -> None:
    """Traverse the dict of strategy spec, update policies needed.

    Args:
        inference_server (InferenceWorkerSet): The instance of inference server, it is actually a ray.ObjectRef.
        force (bool, optional): Force update or not. Defaults to False.
    """

    while True:
        for strategy_spec in inference_server.strategy_spec_dict.values():
            for spec_policy_id in strategy_spec.policy_ids:
                policy_id = f"{strategy_spec.id}/{policy_id}"
                if policy_id in inference_server.policies:
                    weights = ray.get(
                        inference_server.parameter_server.get_weights.remote(
                            spec_id=strategy_spec.id, spec_policy_id=spec_policy_id
                        )
                    )
                    if weights is not None:
                        inference_server.policies[policy_id].load_state_dict(weights)
            time.sleep(1)


def _merge_meta_data(source: Dict[str, Any], dest: Dict[str, Any]):
    # assert
    assert dest["evaluate"] == source["evaluate"], (
        dest["evaluate"],
        source["evaluate"],
    )
    # merge runtime configs, update the environment ids
    dest_environment_ids = dest.get("environment_ids", [])
    dest_environment_ids.extend(source["environment_ids"])
    dest["environment_ids"] = dest_environment_ids


def _merge_data(
    offset: int, source: Dict[str, np.ndarray], dest: Dict[str, np.ndarray]
) -> int:
    """Merge data and return the newest offset.

    Args:
        offset (int): Started offset.
        source (Dict[str, np.ndarray]): Source dict of data.
        dest (Dict[str, np.ndarrray]): Destination dict of data.

    Returns:
        int: Updated offset.
    """

    new_offset = source[Episode.CUR_OBS].shape[0] + offset
    for k, v in dest.items():
        data = source[k]
        if isinstance(v, (np.ndarray, list)):
            # print("shape and key:", k, v.shape, data.shape)
            v[offset : offset + data.shape[0]] = data
        elif v is None and data is None:
            continue
        else:
            raise TypeError(f"Unexpected data type: {type(v)} for key={k}")
        assert new_offset == offset + data.shape[0], (offset, data.shape[0])

    return new_offset


def merge_data_frames(
    policy_refer_dataframes: Dict[PolicyID, List[DataFrame]],
    preset_meta_data: Dict[str, Any],
) -> Dict[AgentID, DataFrame]:
    """Merge dataframes by policy id.

    Args:
        policy_refer_dataframes (Dict[PolicyID, List[DataFrame]]): A dict of dataframe lists, mapping from policy id to dataframe lists.
        preset_meta_data (Dict[str, Any]): Preset meta data.

    Returns:
        Dict[AgentID, DataFrame]: A dict of merged dataframes, mapping from policy ids to dataframes.
    """

    # merge data shapes by pid

    # merge data frames by strategy policy id
    rets: Dict[PolicyID, DataFrame] = {}

    # we group data frames by policy id.
    for policy_id, dataframes in policy_refer_dataframes.items():
        offset = 0
        batch_size = 0
        inner_shapes = dataframes[0].meta_data["data_shapes"]
        placeholder = {}

        for dataframe in dataframes:
            # check consistency
            for k, v in dataframe.meta_data["data_shapes"].items():
                assert k in inner_shapes, (k, inner_shapes)
                assert v == inner_shapes[k], (k, v, inner_shapes[k])
            batch_size += len(dataframe.meta_data["environment_ids"])

        for k, v in inner_shapes.items():
            if v is None:
                placeholder[k] = None
            else:
                placeholder[k] = np.zeros((batch_size,) + v)

        rets[policy_id] = DataFrame(
            identifier=None, data=placeholder, meta_data=preset_meta_data.copy()
        )

        for dataframe in dataframes:
            # stack data
            offset = _merge_data(offset, dataframe.data, rets[policy_id].data)
            # merge runtime configs
            _merge_meta_data(dataframe.meta_data, rets[policy_id].meta_data)

    return rets


def unmerge_policy_rets(
    client_responses: Dict[str, List[DataFrame]],
    rets: Dict[str, np.ndarray],
    client_agent_seg_tups: List[Tuple[str, AgentID, int]],
    meta_data_buffers: Dict[str, Dict[AgentID, Dict]],
):
    """Split policy outputs by clients and agents.

    Args:
        client_responses (Dict[str, List[DataFrame]]): A dict of agent dataframes, a placeholder.
        rets (Dict[str, np.ndarray]): A dict of policy outputs, stacked.
        client_agent_seg_tups (List[Tuple[str, AgentID, int]]): A list of 3-tuples, i.e., (client_id, agent_id, segment_length).
        meta_data_buffers (Dict[str, Dict[AgentID, Dict]]): Runtime buffers, mapping from client id to dicts of dataframe runtime configurations.
    """

    start = 0
    for client_id, agent_id, seg_size in client_agent_seg_tups:
        if client_id not in client_responses:
            client_responses[client_id] = []
        segment_data = {}
        for k, v in rets.items():
            if v is None:
                segment_data[k] = None
            else:
                segment_data[k] = v[start : start + seg_size]
        client_responses[client_id].append(
            DataFrame(
                identifier=agent_id,
                data=segment_data,
                meta_data=meta_data_buffers[client_id][agent_id],
            )
        )
        start += seg_size
