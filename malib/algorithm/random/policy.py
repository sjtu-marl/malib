import numpy as np
import torch
import gym

from malib.algorithm.common.policy import Policy
from malib.utils.episode import Episode
from malib.utils.typing import DataTransferType, Any, List, Tuple
from malib.algorithm.common.model import get_model


class RandomPolicy(Policy):
    def __init__(
        self,
        registered_name,
        observation_space,
        action_space,
        model_config,
        custom_config,
        **kwargs
    ):
        super().__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
            **kwargs
        )

        self.set_actor(
            get_model(self.model_config["actor"])(observation_space, action_space)
        )
        self.set_critic(
            get_model(self.model_config["critic"])(
                observation_space, gym.spaces.Discrete(1)
            )
        )

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        raise NotImplementedError

    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[DataTransferType, DataTransferType, List[DataTransferType]]:
        actor_rnn_state, critic_rnn_state = kwargs[Episode.RNN_STATE]
        # assert len(actor_rnn_state) == len(critic_rnn_state) == len(observation)
        logits = torch.softmax(self.actor(observation), dim=-1)
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_prob = torch.zeros((len(observation), self.action_space.n)).numpy()
        elif isinstance(self.action_space, gym.spaces.Box):
            action_prob = np.random.random(len(observation))
        if "legal_moves" in kwargs:
            mask = torch.zeros_like(logits)
            mask[kwargs["legal_moves"]] = 1
        elif kwargs.get("action_mask") is not None:
            mask = torch.FloatTensor(kwargs["action_mask"])
        else:
            mask = torch.ones_like(logits)
        logits = mask * logits
        if isinstance(self.action_space, gym.spaces.Discrete):
            if len(logits.shape) > 1:
                action = logits.argmax(dim=-1).numpy()
            else:
                action = logits.detach().numpy()
        else:
            action = (
                torch.distributions.Normal(logits[:, 0], logits[:, 1]).sample().numpy()
            )
        return action, action_prob, (actor_rnn_state, critic_rnn_state)

    def get_initial_state(self, batch_size: int = None) -> List[DataTransferType]:
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        shape += (1,)
        return [
            # represent general actor and critic rnn states
            np.zeros(shape)
            for _ in range(2)
        ]

    def value_function(self, *args, **kwargs):
        shape = kwargs[Episode.CUR_OBS].shape
        return np.zeros(shape=(shape[0], 1))

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def set_weights(self, parameters):
        pass

    def reset(self):
        pass
