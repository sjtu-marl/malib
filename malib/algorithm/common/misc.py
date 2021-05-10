import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

from malib.utils.typing import Dict, Any, List, Union, DataTransferType
from malib.backend.datapool.offline_dataset_server import Episode


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


def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """

    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits: DataTransferType, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb

    :param DataTransferType logits: Unnormalized log-probs.
    :param float temperature: Non-negative scalar.
    :param bool hard: If ture take argmax, but differentiate w.r.t. soft sample y
    :returns [batch_size, n_class] sample from the Gumbel-Softmax distribution. If hard=True, then the returned sample
        will be one-hot, otherwise it will be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def cumulative_td_errors(
    start: int, end: int, offset: int, value, td_errors, ratios, gamma: float
):
    v = np.zeros_like(value)
    assert end - offset > start, (start, end, offset)
    for s in range(start, end - offset):
        pi_of_c = 1.0
        trace_errors = [td_errors[s]]
        for t in range(s + 1, s + offset):
            pi_of_c *= ratios[t - 1]
            trace_errors.append(gamma ** (t - start) * pi_of_c * td_errors[t])
        v[s] = value[s] + np.sum(trace_errors)
    return v


def v_trace(
    policy: "Policy", batch: Dict[str, Any], ratio_clip: float = 1e3
) -> Dict[str, Any]:
    """Implementation for V-trace (https://arxiv.org/abs/1802.01561)

    :param policy: Policy, policy instance
    :param batch: Dict[str, Any], batch
    :param ratio_clip: float, ratio clipping value
    :return: return new batch with V-trace target
    """

    # compute importance sampling along the horizon
    old_policy_dist = batch[Episode.ACTION_DIST]
    old_action_dist = old_policy_dist[batch[Episode.ACTIONS]]
    cur_dist = policy.actor()(batch[Episode.CUR_OBS])
    cur_action_dist = cur_dist[batch[Episode.ACTIONS]]

    # NOTE(ming): we should avoid zero division here
    clipped_is_ratio = np.minimum(ratio_clip, cur_action_dist / old_action_dist)
    # calculate new state value
    state_values = batch[Episode.STATE_VALUE]
    rewards = batch[Episode.REWARDS]
    dones = batch[Episode.DONES]

    # ignore the last one state value?
    td_errors = np.zeros_like(rewards)
    td_errors[:-1] = (
        rewards[:-1] + policy.config["gamma"] * state_values[1:] - state_values[:-1]
    )
    terminal_state_value = policy.critic()(batch[Episode.NEXT_OBS][-1])
    # we support infinite episode mode
    td_errors[-1] = (
        rewards[-1]
        + policy.config["gamma"] * terminal_state_value * dones[-1]
        - state_values[-1]
    )
    discounted_td_errors = clipped_is_ratio * td_errors

    batch[Episode.STATE_VALUE] = cumulative_td_errors(
        start=0,
        end=len(rewards),
        offset=1,
        value=state_values,
        td_errors=discounted_td_errors,
        ratios=clipped_is_ratio,
        gamma=policy.config["gamma"],
    )

    return batch


def non_centered_rmsprop(
    gradient: Union[torch.Tensor, DataTransferType],
    delta: Union[torch.Tensor, DataTransferType],
    alpha: float,
    eta: float,
    eps: float,
):
    """Implementation of non-centered RMSProb algorithm (# TODO(ming): add reference here)

    :param gradient: Union[torch.Tensor, DataTransferType], bootstrapped gradient
    :param delta: Union[torch.Tensor, DataTransferType]
    :param alpha: float, moving factor
    :param eta: flat, learning step
    :param eps: float, control exploration
    :return:
    """

    gradient = alpha * gradient + (1.0 - alpha) * delta ** 2
    delta = -eta * delta / np.sqrt(gradient + eps)
    return delta


class GradientOps:
    @staticmethod
    def add(source: Dict, delta: Dict):
        for k, v in delta.items():
            if isinstance(v, Dict):
                source[k] = GradientOps.add(source[k], v)
            else:  # if isinstance(v, DataTransferType):
                assert source[k].data.shape == v.shape, (source[k].data.shape, v.shape)
                source[k].data = source[k].data + v  # v
            # else:
            #     raise errors.UnexpectedType(f"unexpected gradient type: {type(v)}")
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
        else:
            res = np.mean(gradients, axis=0)
            return res

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
        else:  # if isinstance(gradients[0], DataTransferType):
            res = np.sum(gradients, axis=0)
            return res


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
