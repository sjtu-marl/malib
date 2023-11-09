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

from typing import Any, List, Dict, Type, Tuple
from functools import reduce
from operator import mul
from collections import namedtuple

import os

import gym
import torch
import numpy as np

from malib.remote.interface import RemoteInterface
from malib.utils.typing import AgentID, DataFrame
from malib.utils.timing import Timing
from malib.utils.episode import Episode
from malib.common.strategy_spec import StrategySpec

from malib.models.config import ModelConfig
from malib.rl.common.policy import Policy, PolicyReturn


Connection = namedtuple("Connection", "sender,recver,runtime_config,rnn_states")
PolicyReturnWithObs = namedtuple("PolicyReturnWithObs", PolicyReturn._fields + ("obs",))


class InferenceClient(RemoteInterface):
    def __init__(
        self,
        entry_point: str,
        policy_cls: Type,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config: ModelConfig,
    ) -> None:
        """Create ray-based inference server.

        Args:
            entry_point (str): Entrypoint for model update.
            observation_space (gym.Space): Observation space related to the governed environment agents.
            action_space (gym.Space): Action space related to the governed environment agents.
        """

        self.observation_space = observation_space
        self.action_space = action_space

        self.fixed_policy: Policy = policy_cls(
            observation_space, action_space, model_config
        )
        self.active_policy: Policy = policy_cls(
            observation_space,
            action_space,
            model_config,
            model_entry_point=entry_point,
        )

    def shutdown(self):
        pass

    def process_obs(self, raw_observation: Any) -> np.ndarray:
        return self.fixed_policy.preprocessor.transform(raw_observation)

    def compute_action(
        self,
        raw_obs: Any,
        state: Any,
        last_reward: float,
        last_done: float,
        active_policy: bool = False,
        checkpoint: str = None,
        require_obs_return: bool = True,
    ) -> PolicyReturnWithObs:
        """Compute actions for given observations.

        Args:
            raw_obs (Any): Raw observations.
            state (Any): State.
            last_reward (float): Last reward.
            last_done (float): Last done.
            active_policy (bool, optional): Whether to use active model. Defaults to False.
            checkpoint (str, optional): Checkpoint path. Defaults to None.

        Returns:
            PolicyReturnWithObs: An instance of PolicyReturnWithObs.
        """

        if active_policy:
            policy = self.active_policy
            evaluate = False
        else:
            policy = self.fixed_policy
            evaluate = True

            if checkpoint is not None:
                if not os.path.exists(checkpoint):
                    raise RuntimeError(f"Checkpoint {checkpoint} not found.")
                policy.model.load_state_dict(torch.load(checkpoint))

        with torch.inference_mode():
            obs = self.fixed_policy.preprocessor.transform(raw_obs)
            obs = torch.from_numpy(obs).float()
            # FIXME(ming): act mask and hidden state is set to None,
            #   not feasible for cases which require them
            policy_return = policy.compute_action(
                observation=obs,
                act_mask=None,
                evaluate=evaluate,
                hidden_state=None,
                state=state,
                last_reward=last_reward,
                last_done=last_done,
            )
            _returns: dict = policy_return._asdict()
            if require_obs_return:
                _returns.update({"obs": obs})
            policy_return = PolicyReturnWithObs(**_returns)
            return policy_return

    def compute_action_with_frames(
        self, dataframes: List[DataFrame], runtime_config: Dict[str, Any]
    ) -> List[DataFrame]:
        timer = Timing()
        strategy_specs: Dict[AgentID, StrategySpec] = runtime_config["strategy_specs"]
        return_dataframes: List[DataFrame] = []

        assert len(dataframes) > 0

        for dataframe in dataframes:
            with timer.time_avg("others"):
                agent_id = dataframe.identifier
                spec = strategy_specs[agent_id]
                batch_size = dataframe.meta_data["env_num"]
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

            with timer.time_avg("compute_action"):
                (
                    rets[Episode.ACTION],
                    rets[Episode.ACTION_LOGITS],
                    rets[Episode.ACTION_DIST],
                    rets[Episode.RNN_STATE],
                ) = policy.compute_action(
                    observation=observation.reshape(batch_size, -1), **kwargs
                )

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
        return return_dataframes


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
