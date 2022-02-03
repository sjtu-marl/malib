"""
Reference: https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch.py
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, input_shape, available_actions_count):
        """Build a conv net for DQN
        input_shape: a list with 3 integers for input_channels, height and width.
        available_actions_count: an integer for number of discrete actions.
        """
        super(DuelQNet, self).__init__()
        assert len(input_shape) == 3, "Input shape should be a list of 3 integers."
        num_channel, h, w = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self._conv_out_dim = self._build_conv_output_shape(num_channel, h, w)

        self.ffn = nn.Linear(self._conv_out_dim, 64)
        self.state_fc = nn.Linear(64, 1)
        self.advantage_fc = nn.Linear(64, available_actions_count)

    def _build_conv_output_shape(self, num_channel, h, w):
        dummy_input = torch.zeros((1, num_channel, h, w))
        x = self.conv1(dummy_input)
        x = self.conv3(self.conv2(x))
        x = self.conv4(x)
        x = x.view(1, -1)
        return x.shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # XXX(ziyu): shoud we have relu here?
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.ffn(x))
        state_value = self.state_fc(x).reshape(-1, 1)
        advantage_values = self.advantage_fc(x)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

        return x
