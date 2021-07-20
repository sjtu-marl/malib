import itertools
from copy import deepcopy
from functools import reduce
from operator import mul
import gym

import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Categorical, Normal

from copy import deepcopy
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType, Dict, Tuple, BehaviorMode, Any
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import TrainingMetric

EPS = 1e-5


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class BC(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(BC, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.obs_dim = reduce(mul, self.preprocessor.observation_space.shape)
        self._discrete = isinstance(action_space, gym.spaces.Discrete)
        self._action_dim = action_space.n if self._discrete else action_space.shape[0]

        actor = get_model(model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        # register state handler
        self.set_actor(actor)
        self.register_state(self._actor, "actor")

    def compute_action(self, observation, **kwargs):
        logits = self.actor(observation)

        if self._discrete:
            assert len(logits.shape) > 1, logits.shape
            if "action_mask" in kwargs:
                mask = torch.FloatTensor(kwargs["action_mask"]).to(logits.device)
            else:
                mask = torch.ones_like(logits, device=logits.device, dtype=logits.dtype)
            logits = logits * mask
            assert len(logits.shape) > 1, logits.shape
            m = Categorical(logits=logits)
            probs = m.probs
            actions = torch.argmax(probs, dim=-1, keepdim=True).detach()
        else:
            # raise NotImplementedError
            m = Normal(*logits)
            probs = torch.cat(logits, dim=-1)
            actions = m.sample().detach()

        extra_info = {}
        if self._discrete and mask is not None:
            action_probs = torch.zeros_like(probs, device=probs.device)
            active_indices = mask > 0
            tmp = probs[active_indices].reshape(mask.shape) / torch.sum(
                probs, dim=-1, keepdim=True
            )
            action_probs[active_indices] = tmp.reshape(-1)
        else:
            action_probs = probs

        extra_info["action_probs"] = action_probs.detach().to("cpu").numpy()

        return (
            actions.to("cpu").numpy(),
            action_probs.detach().to("cpu").numpy(),
            extra_info,
        )

    def compute_actions(self, observation, **kwargs):
        logits = self.actor(observation)
        if self._discrete:
            m = Categorical(logits=logits)
        else:
            m = Normal(*logits)
        actions = m.sample()
        return actions

    def state_dict(self):
        return {"policy": self.actor.state_dict()}

    def set_weights(self, parameters):
        self.actor.load_state_dict(parameters["policy"])
