# -*- coding: utf-8 -*-
from torch import nn
import torch
import numpy as np
import scipy.signal


def init_fc_weights(m, init_method, gain=1.0):
    init_method(m.weight.data, gain=gain)
    nn.init.constant_(m.bias.data, 0)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return (e ** 2) / 2


class PopArt(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )
        self.forward_cnt = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        # Make sure input is float32
        self.forward_cnt.add_(1.0)
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        if train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[: self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution"""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):  # [n_rollout*n_agent, ...]
            mask_repeat = masks.repeat(1, self._recurrent_N).unsqueeze(-1)
            hxs = (hxs * mask_repeat).transpose(0, 1).contiguous()
            x, hxs = self.rnn(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            # print("----->", masks.shape)
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 1).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs
                    * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)
                ).contiguous()

                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
