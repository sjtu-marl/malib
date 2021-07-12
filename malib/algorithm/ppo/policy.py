import gym
import torch
import numpy as np
from functools import reduce
from operator import mul

from typing import Dict, Any
from torch.nn import functional as F
from torch.distributions import Categorical, Normal

from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy


def cal_neglogp(logits):
    return F.log_softmax(logits)


class PPO(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(PPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )
        self.gamma = custom_config.get("gamma", 0.98)

        self._obs_dim = reduce(mul, self.preprocessor.observation_space.shape)
        self._discrete = isinstance(action_space, gym.spaces.Discrete)
        self._action_dim = action_space.n if self._discrete else action_space.shape[0]

        actor = get_model(self.model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        self._target_actor = get_model(self.model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        critic = get_model(self.model_config["critic"])(
            observation_space,
            gym.spaces.Discrete(1),
            custom_config.get("use_cuda", False),
        )
        self._target_critic = get_model(self.model_config["critic"])(
            observation_space,
            gym.spaces.Discrete(1),
            custom_config.get("use_cuda", False),
        )

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")
        self.register_state(self._target_critic, "target_critic")
        self.register_state(self._target_actor, "target_actor")

        self.update_target()

        self._action_dist = None

    @property
    def target_actor(self):
        return self._target_actor

    @property
    def target_critic(self):
        return self._target_critic

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

    def compute_advantage(self, batch):
        # td_value - value
        next_value = self.target_critic(batch[Episode.NEXT_OBS].copy())
        td_value = (
            torch.from_numpy(batch[Episode.REWARD].copy())
            + self.gamma
            * (1.0 - torch.from_numpy(batch[Episode.DONE].copy()).float())
            * next_value
        )
        value = self.critic(batch[Episode.CUR_OBS].copy())
        adv = td_value - value
        return adv

    def value_function(self, states):
        values = self.critic(states)
        return values

    def update_target(self):
        self._target_critic.load_state_dict(self.critic.state_dict())
        self._target_actor.load_state_dict(self.actor.state_dict())

    def target_value_function(self, states):
        return self._target_critic(states)

    def export(self, export_format: str):
        raise NotImplementedError

    def train(self):
        pass

    def eval(self):
        pass
