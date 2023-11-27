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

from malib.models.config import ModelConfig
from malib.rl.common.policy import Policy, PolicyReturn


Connection = namedtuple("Connection", "sender,recver,runtime_config,rnn_states")
PolicyReturnWithObs = namedtuple("PolicyReturnWithObs", PolicyReturn._fields + ("obs",))


class InferenceClient(RemoteInterface):
    def __init__(
        self,
        model_entry_point: str,
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
            model_entry_point=model_entry_point,
        )

    def shutdown(self):
        pass

    def process_obs(self, raw_observation: Any) -> np.ndarray:
        """Convert raw environmental observation to array like.

        Args:
            raw_observation (Any): Raw environmental observation.

        Returns:
            np.ndarray: Array-like observation.
        """

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
