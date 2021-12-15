import gym
import torch
import numpy as np
from functools import reduce
from operator import mul

from torch.nn import functional as F
from torch.distributions import Categorical, Normal

from malib.utils.typing import BehaviorMode, Tuple, DataTransferType, Dict, Any
from malib.utils.episode import EpisodeKey
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy
from malib.algorithm.common import misc


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
        self._discrete_action = isinstance(action_space, gym.spaces.Discrete)
        self._action_dim = (
            action_space.n if self._discrete_action else action_space.shape[0]
        )

        actor = get_model(self.model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        critic = get_model(self.model_config["critic"])(
            observation_space,
            gym.spaces.Discrete(1),
            custom_config.get("use_cuda", False),
        )

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")

    def compute_action(self, observation, **kwargs):
        behavior = kwargs.get("behavior_mode", BehaviorMode.EXPLORATION)
        action_mask = kwargs.get("action_mask")
        with torch.no_grad():
            logits = self.actor(observation)
            # gumbel softmax convert to differentiable one-hot
            if self._discrete_action:
                if action_mask is not None:
                    action_mask = torch.FloatTensor(action_mask).to(logits.device)
                    pi = misc.masked_softmax(logits, action_mask)
                else:
                    pi = F.softmax(logits, dim=-1)

                if behavior == BehaviorMode.EXPLORATION:
                    m = Categorical(probs=pi)
                    actions = m.sample()
                else:
                    actions = pi.argmax(-1)
            else:
                m = Normal(*logits)
                pi = torch.cat(logits, dim=-1)
                actions = m.sample().detach()
        # print(f"------ action: {pi.argmax(-1).numpy()} {pi.numpy()}")
        return actions.numpy(), pi.numpy(), kwargs[EpisodeKey.RNN_STATE],

    def compute_actions(self, observation, **kwargs):
        logits = self.actor(observation)
        if self._discrete_action:
            m = Categorical(logits=logits)
        else:
            m = Normal(*logits)
        actions = m.sample()
        return actions

    def compute_advantage(self, batch):
        # td_value - value
        cast = lambda x: torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
        next_value = self.critic(cast(batch[EpisodeKey.NEXT_OBS]))
        td_value = (
            cast(batch[EpisodeKey.REWARD])
            + self.gamma * (1.0 - cast(batch[EpisodeKey.DONE]).float()) * next_value
        )
        value = self.critic(cast(batch[EpisodeKey.CUR_OBS]))
        adv = td_value - value
        return adv

    def value_function(self, states):
        values = self.critic(states)
        return values

    def export(self, export_format: str):
        raise NotImplementedError

    def train(self):
        pass

    def eval(self):
        pass
