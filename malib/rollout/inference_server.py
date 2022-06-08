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

from typing import Any, List, Dict
from functools import reduce
from operator import mul
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import os
import time
import traceback

import pickle as pkl
import ray
import gym

from ray.util.queue import Queue

from malib import settings
from malib.utils.typing import AgentID, DataFrame
from malib.utils.timing import Timing
from malib.utils.episode import Episode
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.common.policy import Policy
from malib.backend.parameter_server import ParameterServer


RuntimeHandler = namedtuple("RuntimeHandler", "sender,recver,runtime_config,rnn_states")


@ray.remote
class InferenceWorkerSet:
    def __init__(
        self,
        agent_id: AgentID,
        observation_space: gym.Space,
        action_space: gym.Space,
        parameter_server: ParameterServer,
        governed_agents: List[AgentID],
        force_weight_update: bool = False,
    ) -> None:
        self.runtime_agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.parameter_server = parameter_server

        self.thread_pool = ThreadPoolExecutor()
        self.governed_agents = governed_agents
        self.policies: Dict[str, Policy] = {}
        self.strategy_spec_dict: Dict[str, StrategySpec] = {}

        self.runtime: Dict[int, RuntimeHandler] = {}
        self.parameter_buffer_lock = Lock()
        self.thread_pool.submit(_update_weights, self, force_weight_update)

    def shutdown(self):
        self.thread_pool.shutdown(wait=True)
        for _runtime in self.runtime.values():
            _runtime.sender.shutdown(True)
            _runtime.recver.shutdown(True)
        self.runtime: Dict[int, RuntimeHandler] = {}

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
        runtime_id: int,
    ):
        """Connect new inference task with given configuration and queues.

        Args:
            queues (List[Queue]): A list of send and recieve queues.
            runtime_config (Dict[str, Any]): Runtime configuration for rollout.
            runtime_id (int): Refered runtime id.
        """

        send_queue, recv_queue = queues
        is_exisiting = runtime_id in self.runtime
        self.runtime[runtime_id] = RuntimeHandler(
            send_queue,
            recv_queue,
            runtime_config,
            {agent: [] for agent in self.governed_agents},
        )

        strategy_spec: StrategySpec = runtime_config["strategy_specs"][
            self.runtime_agent_id
        ]
        self._update_policies(strategy_spec, self.runtime_agent_id)

        if not is_exisiting:
            self.thread_pool.submit(_compute_action, self, runtime_id)

    def _update_policies(self, strategy_spec: StrategySpec, agent_id: AgentID):
        for strategy_spec_pid in strategy_spec.policy_ids:
            policy_id = f"{strategy_spec.id}/{strategy_spec_pid}"
            if policy_id not in self.policies:
                policy = strategy_spec.gen_policy()
                self.policies[policy_id] = policy


def _get_initial_states(self, runtime_id, observation, policy: Policy, identifier):
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
    """Maintain the intermediate states produced by policy, for session each.

    Args:
        runtime_id (int): Runtime id.
        rnn_states (Any): A tuple of states
        identifier (str): Identifier, agent id in general.
    """

    self.runtime[runtime_id].rnn_states[identifier].append(rnn_states)


def _compute_action(self: InferenceWorkerSet, runtime_id: int):
    """Maintain the session of action compute for runtime handler tagged with `runtime_id`.

    Args:
        self (InferenceWorkerSet): The instance of inference server, it is actually a ray.ObjectRef.
        runtime_id (int): Runtime id.

    Raises:
        e: Any expectation.
    """

    timer = Timing()

    try:
        handler = self.runtime[runtime_id]
        runtime_config = handler.runtime_config
        strategy_specs: Dict[AgentID, StrategySpec] = runtime_config["strategy_specs"]

        while True:

            if handler.recver.empty():
                continue

            data_frames: List[DataFrame] = handler.recver.get()
            rets = {}
            send_data_frames = []
            timer.clear()
            # with self.parameter_buffer_lock:
            with timer.add_time("data_frame_iter"):
                for data_frame in data_frames:
                    spec = strategy_specs[data_frame.identifier]
                    spec_policy_id = spec.sample()
                    policy_id = f"{spec.id}/{spec_policy_id}"
                    policy: Policy = self.policies[policy_id]
                    kwargs = {**data_frame.data, **data_frame.runtime_config}
                    batch_size = len(kwargs["environment_ids"])
                    assert Episode.CUR_OBS in kwargs, kwargs.keys()
                    observation = kwargs.pop(Episode.CUR_OBS)
                    kwargs[Episode.RNN_STATE] = _get_initial_states(
                        self,
                        runtime_id,
                        observation,
                        policy,
                        identifier=data_frame.identifier,
                    )
                    with timer.add_time("compute_action"):
                        (
                            rets[Episode.ACTION],
                            rets[Episode.ACTION_LOGITS],
                            rets[Episode.ACTION_DIST],
                            rets[Episode.RNN_STATE],
                        ) = policy.compute_action(observation=observation, **kwargs)
                    # compute state value
                    with timer.add_time("compute_value"):
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
                        rets[Episode.RNN_STATE],
                        identifier=data_frame.identifier,
                    )
                    # TODO(ming): considering use async sending
                    # handler.sender.put_nowait(send_df)
            handler.sender.put_nowait(send_data_frames)
    except Exception as e:
        traceback.print_exc()
        raise e


def _update_weights(inference_server: InferenceWorkerSet, force: bool = False) -> None:
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
