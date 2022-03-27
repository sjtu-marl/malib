import traceback
import gym
import torch
import numpy as np
import copy

from functools import reduce
from operator import mul

from torch.nn import functional as F
from torch.distributions import Categorical, Normal

from malib.utils.typing import BehaviorMode, Tuple, DataTransferType, Dict, Any, List
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
        **kwargs
    ):
        """Initialize a PPO policy.

        :param registered_name: _description_
        :type registered_name: str
        :param observation_space: _description_
        :type observation_space: gym.spaces.Space
        :param action_space: _description_
        :type action_space: gym.spaces.Space
        :param model_config: _description_, defaults to None
        :type model_config: Dict[str, Any], optional
        :param custom_config: _description_, defaults to None
        :type custom_config: Dict[str, Any], optional
        """

        super(PPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self._discrete_action = isinstance(action_space, gym.spaces.Discrete)

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

    def get_initial_state(self, batch_size: int = None) -> List[DataTransferType]:
        return self.actor.get_initial_state(batch_size)

    def to_device(self, device):
        raise NotImplementedError

    def compute_action(self, observation, **kwargs):
        try:
            behavior = kwargs.get("behavior_mode", BehaviorMode.EXPLORATION)
            action_mask = kwargs.get("action_mask")

            with torch.no_grad():
                logits = self.actor(observation)
                # gumbel softmax convert to differentiable one-hot
                if self._discrete_action:
                    if action_mask is not None:
                        action_mask = torch.FloatTensor(action_mask.copy()).to(
                            logits.device
                        )
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
            return (
                actions.numpy(),
                pi.numpy(),
                kwargs[EpisodeKey.RNN_STATE],
            )
        except Exception as e:
            traceback.print_exc()
            raise e

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy._actor = self_copy._actor.to(device)
        self_copy._critic = self_copy._critic.to(device)
        return self_copy

    def compute_actions(self, observation, **kwargs):
        logits = self.actor(observation)
        if self._discrete_action:
            m = Categorical(logits=logits)
        else:
            m = Normal(*logits)
        actions = m.sample()
        return actions

    # def compute_advantage(self, batch: Dict[str, Any]):
    #     # td_value - value
    #     cast = lambda x: torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
    #     next_value = self.critic(cast(batch[EpisodeKey.NEXT_OBS]))
    #     td_value = (
    #         cast(batch[EpisodeKey.REWARD])
    #         + self.gamma * (1.0 - cast(batch[EpisodeKey.DONE]).float()) * next_value
    #     )
    #     value = self.critic(cast(batch[EpisodeKey.CUR_OBS]))
    #     adv = td_value - value
    #     return adv

    def value_function(self, observation, **kwargs):
        values = self.critic(observation)
        return values.detach().numpy()

    def export(self, export_format: str):
        raise NotImplementedError
