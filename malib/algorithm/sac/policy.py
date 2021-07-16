import itertools
from copy import deepcopy
from functools import reduce
from operator import mul

import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.distributions import Categorical
from copy import deepcopy
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import TrainingMetric

EPS = 1e-5


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class SAC(Policy):
    def __init__(
        self,
        registered_name,
        observation_space,
        action_space,
        model_config,
        custom_config,
    ):
        super(SAC, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.gamma = 0.99 if custom_config is None else custom_config.get("gamma", 0.99)
        self.entropy_coef = (
            0.1 if custom_config is None else custom_config.get("entropy_coef", 0.1)
        )
        self.polyak = (
            0.99 if custom_config is None else custom_config.get("polyak", 0.99)
        )
        self.lr = 1e-4 if custom_config is None else custom_config.get("lr", 1e-4)

        self.obs_dim = reduce(mul, self.preprocessor.observation_space.shape)
        self.act_dim = action_space.n
        self.pi = get_model(model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        self.q1 = get_model(model_config["critic"])()(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )
        self.q2 = get_model(model_config["critic"])()(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )

        # self.pi_optimizer = Adam(self.pi.parameters(), lr=self.lr)
        # self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        # self.q_optimizer = Adam(self.q_params, lr=self.lr)

        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p1, p2 in zip(self.q1_targ.parameters(), self.q1_targ.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def update(self, batch):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

        self.update_target()

        loss_names = [
            "pi_loss",
            "q_loss",
        ]
        stats_list = [
            loss_pi.detach().numpy(),
            loss_q.detach().numpy(),
        ]

        return {
            TrainingMetric.LOSS: (loss_pi + loss_q).detach().numpy(),
            **dict(zip(loss_names, stats_list)),
        }

    def compute_loss_q(self, batch):
        r = torch.as_tensor(batch[Episode.REWARDS].copy(), dtype=torch.float32).view(
            -1, 1
        )
        a = torch.LongTensor(batch[Episode.ACTIONS].copy()).view(-1, 1)
        o = batch[Episode.CUR_OBS].copy()
        o2 = batch[Episode.NEXT_OBS].copy()
        d = torch.as_tensor(batch[Episode.DONES].copy(), dtype=torch.float32).view(
            -1, 1
        )

        q1 = self.q1(o).gather(1, a)
        q2 = self.q2(o).gather(1, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target Q-values
            q1_pi_targ = self.q1_targ(o2).max(1)[0]
            q2_pi_targ = self.q2_targ(o2).max(1)[0]
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).unsqueeze(1)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, batch):
        o = batch[Episode.CUR_OBS].copy()
        pi = self.pi(o)
        pi_distribution = Categorical(pi)
        log_pi = torch.log(pi + EPS)
        entropy = pi_distribution.entropy()
        with torch.no_grad():
            q1_pi = self.q1(o)
            q2_pi = self.q2(o)
            q_pi = torch.min(q1_pi, q2_pi)
            v = (pi * q_pi).sum(1).unsqueeze(1).detach()
            adv = q_pi - v

        J = -(log_pi * adv).mean()
        E = -self.entropy_coef * entropy.mean()
        loss_pi = E + J
        return loss_pi

    def update_target(self):
        with torch.no_grad():
            for p1, p2, p_targ_1, p_targ_2 in zip(
                self.q1.parameters(),
                self.q2.parameters(),
                self.q1_targ.parameters(),
                self.q2_targ.parameters(),
            ):
                p_targ_1.data.mul_(self.polyak)
                p_targ_1.data.add_((1 - self.polyak) * p1.data)
                p_targ_2.data.mul_(self.polyak)
                p_targ_2.data.add_((1 - self.polyak) * p2.data)

    def compute_actions(self, observation, **kwargs):
        pass

    def compute_action(self, observation, **kwargs):
        with torch.no_grad():
            pi = torch.softmax(self.pi(np.asarray([observation])), dim=-1)
            if "legal_moves" in kwargs:
                mask = torch.zeros_like(pi)
                mask[:, kwargs["legal_moves"]] = 1
                pi = mask * pi
            elif "action_mask" in kwargs:
                mask = torch.FloatTensor(kwargs["action_mask"])
                pi = mask * pi
            pi = pi / pi.sum()
            a = Categorical(probs=pi).sample()
            extra_info = {"action_probs": pi.view(-1).numpy()}
            return a.item(), None, extra_info

    def state_dict(self):
        return {
            "policy": self.pi.state_dict(),
            "q_1": self.q1.state_dict(),
            "q_1_target": self.q1_targ.state_dict(),
            "q_2": self.q2.state_dict(),
            "q_2_target": self.q2_targ.state_dict(),
        }

    def set_weights(self, parameters):
        self.pi.load_state_dict(parameters["policy"])
        self.q1.load_state_dict(parameters["q_1"])
        self.q1_targ.load_state_dict(parameters["q_1_target"])
        self.q2.load_state_dict(parameters["q_2"])
        self.q2_targ.load_state_dict(parameters["q_2_target"])

    def train(self):
        pass

    def eval(self):
        pass
