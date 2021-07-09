import torch
import torch.nn as nn
import torch.nn.functional as F
from malib.algorithm.common.model import Model


class QMixer(Model):
    def __init__(self, obs_dim, num_agents, model_config=None):
        super(QMixer, self).__init__(obs_dim, num_agents)
        self.n_agents = self.output_dim

        self.embed_dim = (
            32 if model_config is None else model_config.get("mixer_embed_dim", 32)
        )
        self.hyper_hidden_dim = (
            64 if model_config is None else model_config.get("hyper_hidden_dim", 64)
        )

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim * self.n_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim),
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.input_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
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
