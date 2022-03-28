import torch
import numpy as np
import math
import torch

import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions.utils import lazy_property
from torch.distributions import utils as distr_utils
from torch.distributions.categorical import Categorical as TorchCategorical

from malib.utils.typing import Dict, List, DataTransferType, Any, Optional


def soft_update(target, source, tau):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1).

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """

    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, explore: bool = True):
    """Draw a sample from the Gumbel-Softmax distribution.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """

    y = logits
    if explore:
        y += sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits: DataTransferType, temperature=1.0, hard=False, explore=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb

    :param DataTransferType logits: Unnormalized log-probs.
    :param float temperature: Non-negative scalar.
    :param bool hard: If ture take argmax, but differentiate w.r.t. soft sample y
    :returns [batch_size, n_class] sample from the Gumbel-Softmax distribution. If hard=True, then the returned sample
        will be one-hot, otherwise it will be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, explore)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor):
    logits = F.normalize(logits)
    logits = torch.clamp(logits - (1.0 - mask) * 1e9, -1e9, 1e9)
    probs = F.softmax(logits, dim=-1)  # * mask
    # probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)
    Z = probs.sum(dim=-1, keepdim=True)
    return probs / Z


def monte_carlo_discounted(rewards, dones, gamma: float) -> torch.Tensor:
    running_add = 0
    returns = []

    for step in reversed(range(len(rewards))):
        running_add = rewards[step] + (1.0 - dones[step]) * gamma * running_add
        returns.insert(0, running_add)

    return torch.stack(returns)


def temporal_difference(reward, next_value, done, gamma: float) -> torch.Tensor:
    q_values = reward + (1.0 - done) * gamma * next_value
    return q_values


def generalized_advantage_estimation(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
):
    gae = 0
    adv = []

    delta = rewards + (1.0 - dones) * gamma * next_values - values
    for step in reversed(range(len(rewards))):
        gae = delta[step] + (1.0 - dones[step]) * gamma * lam * gae
        adv.insert(0, gae)

    return torch.stack(adv)


def vtrace(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    log_probs: torch.Tensor,
    worker_logprobs: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    gae = 0
    adv = []

    limit = torch.FloatTensor([1.0]).to(values.device)
    ratio = torch.min(limit, (worker_logprobs - log_probs).sum().exp())

    assert rewards.shape == dones.shape == next_values.shape == values.shape, (
        rewards.shape,
        dones.shape,
        next_values.shape,
        values.shape,
    )
    delta = rewards + (1.0 - dones) * gamma * next_values - values
    delta = ratio * delta

    for step in reversed(range(len(rewards))):
        gae = (1.0 - dones[step]) * gamma * lam * gae
        gae = delta[step] + ratio * gae
        adv.insert(0, gae)

    return torch.stack(adv)


class GradientOps:
    @staticmethod
    def add(source: Any, delta: Any):
        """Apply gradients (delta) to parameters (source)"""

        if isinstance(source, Dict) and isinstance(delta, Dict):
            for k, v in delta.items():
                if isinstance(v, Dict):
                    source[k] = GradientOps.add(source[k], v)
                else:  # if isinstance(v, DataTransferType):
                    assert source[k].data.shape == v.shape, (
                        source[k].data.shape,
                        v.shape,
                    )
                    if isinstance(v, np.ndarray):
                        source[k].data.copy_(source[k].data + v)
                    elif isinstance(v, torch.Tensor):
                        source[k].data.copy_(source[k].data + v.data)
                    else:
                        raise TypeError(
                            "Inner type of delta should be numpy.ndarray or torch.Tensor, but `{}` detected".format(
                                type(v)
                            )
                        )
        elif isinstance(source, torch.Tensor):
            if isinstance(delta, torch.Tensor):
                source.data.copy_(source.data + delta.data)
            elif isinstance(delta, np.ndarray):
                source.data.copy_(source.data + delta)
            else:
                raise TypeError("Unexpected delta type: {}".format(type(delta)))
        else:
            raise TypeError(
                "Source data must be a dict or torch tensor but got: {}".format(
                    type(source)
                )
            )
        return source

    @staticmethod
    def mean(gradients: List):
        if len(gradients) < 1:
            return gradients
        if isinstance(gradients[0], dict):
            keys = list(gradients[0].keys())
            res = {}
            for k in keys:
                res[k] = GradientOps.mean([grad[k] for grad in gradients])
            return res
        elif isinstance(gradients[0], np.ndarray):
            res = np.mean(gradients, axis=0)
            return res
        elif isinstance(gradients[0], torch.Tensor):
            raise NotImplementedError(
                "Do not support tensor-based gradients aggragation yet."
            )
        else:
            raise TypeError("Illegal data type: {}".format(type(gradients[0])))

    @staticmethod
    def sum(gradients: List):
        """Sum gradients.

        :param List gradients: A list of gradients.
        :return:
        """

        if len(gradients) < 1:
            return gradients

        if isinstance(gradients[0], dict):
            keys = list(gradients[0].keys())
            res = {}
            for k in keys:
                res[k] = GradientOps.sum([grad[k] for grad in gradients])
            return res
        elif isinstance(
            gradients[0], np.ndarray
        ):  # if isinstance(gradients[0], DataTransferType):
            res = np.sum(gradients, axis=0)
            return res
        elif isinstance(gradients[0], torch.Tensor):
            raise NotImplementedError(
                "Do not support tensor-based gradients aggragation yet."
            )
        else:
            raise TypeError("Illegal data type: {}".format(type(gradients[0])))


class OUNoise:
    """https://github.com/songrotek/DDPG/blob/master/ou_noise.py"""

    def __init__(self, action_dimension: int, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class EPSGreedy:
    def __init__(self, action_dimension: int, threshold: float = 0.3):
        self._action_dim = action_dimension
        self._threshold = threshold


class MaskedCategorical:
    def __init__(self, scores, mask=None):
        self.mask = mask
        if mask is None:
            self.cat_distr = TorchCategorical(F.softmax(scores, dim=-1))
            self.n = scores.shape[0]
            self.log_n = math.log(self.n)
        else:
            self.n = self.mask.sum(dim=-1)
            self.log_n = (self.n + 1e-17).log()
            self.cat_distr = TorchCategorical(
                MaskedCategorical.masked_softmax(scores, self.mask)
            )

    @lazy_property
    def probs(self):
        return self.cat_distr.probs

    @lazy_property
    def logits(self):
        return self.cat_distr.logits

    @lazy_property
    def entropy(self):
        if self.mask is None:
            return self.cat_distr.entropy() * (self.n != 1)
        else:
            entropy = -torch.sum(
                self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1
            )
            does_not_have_one_category = (self.n != 1.0).to(dtype=torch.float32)
            # to make sure that the entropy is precisely zero when there is only one category
            return entropy * does_not_have_one_category

    @lazy_property
    def normalized_entropy(self):
        return self.entropy / (self.log_n + 1e-17)

    def sample(self):
        return self.cat_distr.sample()

    def rsample(self, temperature=None, gumbel_noise=None):
        if gumbel_noise is None:
            with torch.no_grad():
                uniforms = torch.empty_like(self.probs).uniform_()
                uniforms = distr_utils.clamp_probs(uniforms)
                gumbel_noise = -(-uniforms.log()).log()
            # TODO(serhii): This is used for debugging (to get the same samples) and is not differentiable.
            # gumbel_noise = None
            # _sample = self.cat_distr.sample()
            # sample = torch.zeros_like(self.probs)
            # sample.scatter_(-1, _sample[:, None], 1.0)
            # return sample, gumbel_noise

        elif gumbel_noise.shape != self.probs.shape:
            raise ValueError

        if temperature is None:
            with torch.no_grad():
                scores = self.logits + gumbel_noise
                scores = MaskedCategorical.masked_softmax(scores, self.mask)
                sample = torch.zeros_like(scores)
                sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
                return sample, gumbel_noise
        else:
            scores = (self.logits + gumbel_noise) / temperature
            sample = MaskedCategorical.masked_softmax(scores, self.mask)
            return sample, gumbel_noise

    def log_prob(self, value):
        if value.dtype == torch.long:
            if self.mask is None:
                return self.cat_distr.log_prob(value)
            else:
                return self.cat_distr.log_prob(value) * (self.n != 0.0).to(
                    dtype=torch.float32
                )
        else:
            max_values, mv_idxs = value.max(dim=-1)
            relaxed = (max_values - torch.ones_like(max_values)).sum().item() != 0.0
            if relaxed:
                raise ValueError(
                    "The log_prob can't be calculated for the relaxed sample!"
                )
            return self.cat_distr.log_prob(mv_idxs) * (self.n != 0.0).to(
                dtype=torch.float32
            )

    @staticmethod
    def masked_softmax(logits, mask):
        """
        This method will return valid probability distribution for the particular instance if its corresponding row
        in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
        This is just a technical workaround that allows `Categorical` class usage.
        If probs doesn't sum to one there will be an exception during sampling.
        """
        probs = F.softmax(logits, dim=-1) * mask
        probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)
        Z = probs.sum(dim=-1, keepdim=True)
        return probs / Z
