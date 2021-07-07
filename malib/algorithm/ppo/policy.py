import gym
import torch
from functools import reduce
from operator import mul

from typing import Dict, Any
from torch.nn import functional as F
from torch.distributions import Categorical

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
        probs = self.actor(observation).detach()
        mask = None  # mask not always exists
        if "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"]).to(probs.device)
            mask = mask.long().unsqueeze(0)
            probs = mask * probs
        action = probs.argmax(dim=-1).view((-1, 1))

        extra_info = {}
        action_prob = torch.zeros_like(probs, device=probs.device)
        if mask is not None:
            active_indices = mask > 0
            action_prob[active_indices] = probs[active_indices] / probs.sum()
        extra_info["action_probs"] = action_prob.numpy()

        return action.view((-1,)).numpy(), action_prob.numpy(), extra_info

    def compute_actions(self, observation, **kwargs):
        probs = self.actor(observation)
        m = Categorical(probs=probs)
        actions = m.sample()
        return actions

    def compute_advantage(self, batch):
        # td_value - value
        next_value = self.target_critic(batch[Episode.NEXT_OBS].copy())
        td_value = (
            torch.from_numpy(batch[Episode.REWARDS].copy())
            + self.gamma
            * (1.0 - torch.from_numpy(batch[Episode.DONES].copy()).float())
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
