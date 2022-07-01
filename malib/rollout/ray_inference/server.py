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

import os

import pickle as pkl
import ray
import gym

from malib import settings
from malib.remote.interface import RemoteInterface
from malib.utils.typing import AgentID, DataFrame
from malib.utils.timing import Timing
from malib.utils.episode import Episode
from malib.common.strategy_spec import StrategySpec
from malib.algorithm.common.policy import Policy
from malib.backend.parameter_server import ParameterServer


ClientHandler = namedtuple("ClientHandler", "sender,recver,runtime_config,rnn_states")


class RayInferenceWorkerSet(RemoteInterface):
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
        self.policy_version: Dict[str, int] = {}
        self.strategy_spec_dict: Dict[str, StrategySpec] = {}

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

    def compute_action(
        self, dataframes: List[DataFrame], runtime_config: Dict[str, Any]
    ) -> List[DataFrame]:
        timer = Timing()
        strategy_specs: Dict[AgentID, StrategySpec] = runtime_config["strategy_specs"]
        return_dataframes: List[DataFrame] = []

        # check policy
        self._update_policies(
            runtime_config["strategy_specs"][self.runtime_agent_id],
            self.runtime_agent_id,
        )

        for dataframe in dataframes:
            with timer.time_avg("others"):
                agent_id = dataframe.identifier
                spec = strategy_specs[agent_id]
                batch_size = len(dataframe.meta_data["environment_ids"])
                spec_policy_id = spec.sample()
                policy_id = f"{spec.id}/{spec_policy_id}"
                policy: Policy = self.policies[policy_id]
                kwargs = {
                    Episode.DONE: dataframe.data[Episode.DONE],
                    Episode.ACTION_MASK: dataframe.data[Episode.ACTION_MASK],
                    "evaluate": dataframe.meta_data["evaluate"],
                }
                observation = dataframe.data[Episode.CUR_OBS]
                kwargs[Episode.RNN_STATE] = _get_initial_states(
                    self,
                    None,
                    observation,
                    policy,
                    identifier=dataframe.identifier,
                )

                rets = {}

            with timer.time_avg("policy_update"):
                info = ray.get(
                    self.parameter_server.get_weights.remote(
                        spec_id=spec.id,
                        spec_policy_id=spec_policy_id,
                        cur_version=self.policy_version[policy_id],
                    )
                )
                if info["weights"] is not None:
                    self.policies[policy_id].load_state_dict(info["weights"])
                    self.policy_version[policy_id] = info["version"] + 1

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

            with timer.time_avg("tail_handler"):
                for k, v in rets.items():
                    if k == Episode.RNN_STATE:
                        continue
                    if len(v.shape) < 1:
                        rets[k] = v.reshape(-1)
                    elif v.shape[0] == 1:
                        continue
                    else:
                        rets[k] = v.reshape(batch_size, -1)
            return_dataframes.append(
                DataFrame(identifier=agent_id, data=rets, meta_data=dataframe.meta_data)
            )
        # print(f"timer information: {timer.todict()}")
        return return_dataframes

    def _update_policies(self, strategy_spec: StrategySpec, agent_id: AgentID):
        for strategy_spec_pid in strategy_spec.policy_ids:
            policy_id = f"{strategy_spec.id}/{strategy_spec_pid}"
            if policy_id not in self.policies:
                policy = strategy_spec.gen_policy()
                self.policies[policy_id] = policy
                self.policy_version[policy_id] = -1


def _get_initial_states(self, client_id, observation, policy: Policy, identifier):
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


def _update_weights(
    inference_server: RayInferenceWorkerSet, force: bool = False
) -> None:
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
                        inference_server.policies[policy_id].load_state_dict(
                            weights["weights"]
                        )
            # time.sleep(1)
