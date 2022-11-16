# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, List, Any, Optional, Type, Sequence

import math
import torch
import numpy as np

import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions.utils import lazy_property
from torch.distributions import utils as distr_utils
from torch.distributions.categorical import Categorical as TorchCategorical


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """Perform soft update.

    Args:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float): Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def onehot_from_logits(logits: torch.Tensor, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"the logits should be an instance of `torch.Tensor`, while the given type is {type(logits)}"
        )

    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps should locate in [0, 1], while the given value is {eps}")

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

    is_random = (torch.rand(logits.shape[0]) <= eps).float().reshape(-1, 1)

    assert len(rand_acs.shape) == len(argmax_acs.shape) == len(is_random.shape), (
        rand_acs.shape,
        argmax_acs.shape,
        is_random.shape,
    )

    return (1 - is_random) * argmax_acs + is_random * rand_acs
    # chooses between best and random actions using epsilon greedy


def sample_gumbel(
    shape: torch.Size, eps: float = 1e-20, tens_type: Type = torch.FloatTensor
) -> torch.Tensor:
    """Sample noise from an uniform distribution withe a given shape. Note the returned tensor is deactivated for gradients computation.

    Args:
        shape (torch.Size): Target shape.
        eps (float, optional): Tolerance to avoid NaN. Defaults to 1e-20.
        tens_type (Type, optional): Indicates the data type of the sampled noise. Defaults to torch.FloatTensor.

    Returns:
        torch.Tensor: A tensor as sampled noise.
    """

    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)

    # U + eps to avoid raising NaN error
    return -torch.log(-torch.log(U + eps) + eps)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor):
    if mask is not None:
        assert isinstance(mask, torch.Tensor), type(mask)
        assert mask.shape == logits.shape, (mask.shape, logits.shape)
        logits = torch.clamp(logits - (1.0 - mask) * 1e9, -1e9, 1e9)
    return logits


def softmax(
    logits: torch.Tensor,
    temperature: float,
    mask: torch.Tensor = None,
    explore: bool = True,
) -> torch.Tensor:
    """Apply softmax to the given logits. With distribution density control and optional exploration noise.

    Args:
        logits (torch.Tensor): Logits tensor.
        temperature (float): Temperature controls the distribution density.
        mask (torch.Tensor, optional): Applying action mask if not None. Defaults to None.
        explore (bool, optional): Add noise to the generated distribution or not. Defaults to True.

    Raises:
        TypeError: Logits should be a `torch.Tensor`.

    Returns:
        torch.Tensor: softmax tensor, shaped as (batch_size, n_classes).
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"logits should be a `torch.Tensor`, while the given is {type(logits)}"
        )

    logits = logits / temperature

    if explore:
        logits = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))

    logits = masked_logits(logits, mask)

    return F.softmax(logits, dim=-1)


def gumbel_softmax(
    logits: torch.Tensor, temperature=1.0, mask: torch.Tensor = None, explore=False
) -> torch.Tensor:
    """Convert a softmax to one hot but gradients computation will be kept.

    Args:
        logits (torch.Tensor): Raw logits tensor.
        temperature (float, optional): Temperature to control the distribution density. Defaults to 1.0.
        mask (torch.Tensor, optional): Action masking. Defaults to None.
        explore (bool, optional): Enable noise adding or not. Defaults to True.

    Returns:
        torch.Tensor: Genearted gumbel softmax, shaped as (batch_size, n_classes)
    """

    y = softmax(logits, temperature, mask, explore)
    y_hard = onehot_from_logits(y)
    y = (y_hard - y).detach() + y
    return y
