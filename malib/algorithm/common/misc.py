import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

from malib.utils.typing import Dict, List, DataTransferType, Any


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
    probs = F.softmax(logits, dim=-1) * mask
    probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)
    Z = probs.sum(dim=-1, keepdim=True)
    return probs / Z


# def non_centered_rmsprop(
#     gradient: Union[torch.Tensor, DataTransferType],
#     delta: Union[torch.Tensor, DataTransferType],
#     alpha: float,
#     eta: float,
#     eps: float,
# ):
#     """Implementation of non-centered RMSProb algorithm (# TODO(ming): add reference here)

#     :param gradient: Union[torch.Tensor, DataTransferType], bootstrapped gradient
#     :param delta: Union[torch.Tensor, DataTransferType]
#     :param alpha: float, moving factor
#     :param eta: flat, learning step
#     :param eps: float, control exploration
#     :return:
#     """

#     gradient = alpha * gradient + (1.0 - alpha) * delta ** 2
#     delta = -eta * delta / np.sqrt(gradient + eps)
#     return delta


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
