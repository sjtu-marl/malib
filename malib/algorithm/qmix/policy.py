import numpy as np
import gym
import torch
from typing import Dict, Any
from copy import deepcopy
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy


class QMIX(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
    ):
        super(QMIX, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )
        self.eps_min = (
            1e-2 if custom_config is None else custom_config.get("eps_min", 1e-2)
        )
        self.eps_max = (
            1.0 if custom_config is None else custom_config.get("eps_max", 1.0)
        )
        self.eps_decay = (
            2000 if custom_config is None else custom_config.get("eps_decay", 2000)
        )
        self.polyak = (
            0.99 if custom_config is None else custom_config.get("polyak", 0.99)
        )
        self.step = 0

        self.obs_dim = self.preprocessor.size
        self.act_dim = action_space.n

        self.q = get_model(model_config["critic"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        self.q_targ = deepcopy(self.q)

        self.set_critic(self.q)

        self.register_state(self.q, "critic")
        self.register_state(self.q_targ, "target_critic")
        self.register_state(self.step, "step")

    def compute_actions(self, observation, **kwargs):
        pass

    def _calc_eps(self):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.step / self.eps_decay
        )

    def compute_action(self, observation, **kwargs):
        self.step += 1
        if np.random.random() < self._calc_eps():
            actions = range(self.action_space.n)
            if "legal_moves" in kwargs:
                actions = kwargs["legal_moves"]
            elif "action_mask" in kwargs:
                actions = np.where(kwargs["action_mask"] == 1)[0]
            action = np.random.choice(actions)
            action_prob = torch.zeros(self.action_space.n)
            action_prob[action] = 1.0
            return action, None, {"action_probs": action_prob}
        probs = torch.softmax(self.q(observation), dim=-1)
        if "legal_moves" in kwargs:
            mask = torch.zeros_like(probs)
            mask[kwargs["legal_moves"]] = 1
            probs = mask * probs
        elif "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"])
            probs = mask * probs
        # probs = probs / probs.sum()
        # action = Categorical(probs=probs).sample()
        action = probs.argmax().view(1)

        extra_info = {"action_probs": probs.detach().numpy()}
        return action.item(), probs.detach().numpy(), extra_info

    def compute_q(self, obs):
        return self.q(obs)

    def compute_target_q(self, obs):
        return self.q_targ(obs)

    def get_parameters(self):
        return self.q.parameters()

    def update_target(self):
        # self.q_targ.load_state_dict(self.q.state_dict())
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
