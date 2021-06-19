"""
Model factory. Add more description
"""

import copy

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from malib.utils.preprocessor import get_preprocessor
from malib.utils.typing import Dict, Any


def mlp(layers_config):
    layers = []
    for j in range(len(layers_config) - 1):
        tmp = [nn.Linear(layers_config[j]["units"], layers_config[j + 1]["units"])]
        if layers_config[j + 1].get("activation"):
            tmp.append(getattr(torch.nn, layers_config[j + 1]["activation"])())
        layers += tmp
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
    ):
        super(MLP, self).__init__()

        obs_dim = get_preprocessor(observation_space)(observation_space).size
        act_dim = get_preprocessor(action_space)(action_space).size
        layers_config: list = (
            self._default_layers()
            if model_config.get("layers") is None
            else model_config["layers"]
        )
        layers_config.insert(0, {"units": obs_dim})
        layers_config.append(
            {"units": act_dim, "activation": model_config["output"]["activation"]}
        )
        self.net = mlp(layers_config)

    def _default_layers(self):
        return [
            {"units": 256, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ]

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        pi = self.net(obs)
        return pi


class RNN(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
    ):
        super(RNN, self).__init__()
        self.hidden_dims = (
            64 if model_config is None else model_config.get("rnn_hidden_dim", 64)
        )

        # default by flatten
        obs_dim = get_preprocessor(observation_space)(observation_space).size()
        act_dim = get_preprocessor(action_space)(action_space).size()

        self.fc1 = nn.Linear(obs_dim, self.hidden_dims)
        self.rnn = nn.GRUCell(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, act_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dims).zero_()

    def forward(self, obs, hidden_state):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dims)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    def __init__(self, obs_dim, num_agents, model_config):
        super(QMixer, self).__init__()
        self.n_agents = num_agents

        self.embed_dim = (
            32 if model_config is None else model_config.get("mixer_embed_dim", 32)
        )
        self.hyper_hidden_dim = (
            64 if model_config is None else model_config.get("hyper_hidden_dim", 64)
        )

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(obs_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim * num_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(obs_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim),
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(obs_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(obs_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, obs):
        bs = agent_qs.size(0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        agent_qs = torch.as_tensor(agent_qs, dtype=torch.float32)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(obs))
        b1 = self.hyper_b_1(obs)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(obs))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(obs).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1)
        return q_tot


def get_model(model_config: Dict[str, Any]):
    model_type = model_config["network"]

    if model_type == "mlp":
        handler = MLP
    elif model_type == "rnn":
        handler = RNN
    elif model_type == "cnn":
        raise NotImplementedError
    elif model_type == "rcnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    def builder(observation_space, action_space, use_cuda=False):
        model = handler(observation_space, action_space, copy.deepcopy(model_config))
        if use_cuda:
            model.cuda()
        return model

    return builder
