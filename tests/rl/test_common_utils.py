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

from typing import Dict, Sequence

import pytest
import numpy as np
import torch

from malib.rl.common import misc


class OrderingNet(torch.nn.Module):
    def __init__(self, np_dict: Dict[str, np.ndarray]):
        super(OrderingNet, self).__init__()
        parameters = {}
        for k, v in np_dict.items():
            parameters[k] = torch.nn.Parameter(data=torch.as_tensor(v).float())
        self.model = torch.nn.ParameterDict(parameters)

    def forward(self, x):
        return x


@pytest.mark.parametrize("tau", [0.1, 0.5, 1])
def test_soft_update(tau):
    np_source_parameters = {"a": np.arange(10), "b": np.arange(20), "c": np.arange(30)}
    np_target_parameters = {"a": np.arange(10), "b": np.arange(20), "c": np.arange(30)}
    mixed_target_parameters = {
        k: np_source_parameters[k] * tau + np_target_parameters[k] * (1 - tau)
        for k in np_target_parameters
    }
    np_shape_error_target_parameters = {
        "a": np.arange(10),
        "b": np.arange(10),
        "c": np.arange(10),
    }

    source = OrderingNet(np_source_parameters)
    target = OrderingNet(np_target_parameters)
    target_shape_error = OrderingNet(np_shape_error_target_parameters)

    with pytest.raises((ValueError, RuntimeError)):
        misc.soft_update(target_shape_error, source, tau)

    misc.soft_update(target, source, tau)
    # checkout target value
    for k, v in target.named_parameters():
        assert np.all(
            np.isclose(v.detach().numpy(), mixed_target_parameters[k.split(".")[-1]])
        )


@pytest.mark.parametrize("eps", [-0.1, 0.0, 0.5, 1.0, 1.2])
def test_onehot_from_logits(eps: float):
    """convert a logits to an one-hot vector"""

    logits = np.random.random(64 * 10).reshape(64, 10)
    logits_tensor = torch.as_tensor(logits)

    with pytest.raises(TypeError):
        misc.onehot_from_logits(logits, eps)

    if 0 <= eps <= 1:
        outputs = misc.onehot_from_logits(logits_tensor, eps)
        assert torch.all(~torch.isnan(outputs)), torch.isnan(outputs)
    else:
        with pytest.raises(ValueError) as e_info:
            misc.onehot_from_logits(logits_tensor, eps)


@pytest.mark.parametrize("eps", [0.0, 1e-20, 1e-10])
@pytest.mark.parametrize("shape", [(1, 2, 3), (1,), (1, 2)])
def test_sample_gumbel(eps: float, shape: Sequence[int]):
    misc.sample_gumbel(shape=torch.Size(shape), eps=eps)


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("explore", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_gumbel_softmax_and_sample(temperature: float, explore: bool, use_mask: bool):
    logits = np.random.random(2 * 10).reshape(2, 10).astype(np.float32)

    with pytest.raises(TypeError) as e_info:
        misc.softmax(logits, temperature, explore)

    pred = misc.softmax(torch.as_tensor(logits).float(), temperature, explore=explore)

    if explore:
        assert logits.shape == pred.shape, (logits.shape, pred.shape)
    else:
        label = np.exp(logits / temperature) / np.exp(logits / temperature).sum(
            -1, keepdims=True
        )
        assert np.all(np.isclose(label, pred.detach().numpy())), (
            label,
            pred.detach().numpy(),
        )

    # test gumbel softmax
    logits = torch.as_tensor(logits).float()
    logits.requires_grad = True

    if use_mask:
        mask = torch.ones_like(logits)
        mask[:, -4:-1] = 1
    else:
        mask = None
    one_hots = misc.gumbel_softmax(logits, temperature, mask, explore)
    idx = torch.where(one_hots == 1)
    assert len(idx[1].shape) == 1
    assert one_hots.requires_grad
