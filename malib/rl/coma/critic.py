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

from typing import Union, Dict, Any, Tuple

import numpy as np
import torch
import gym

from torch import nn
from gym import spaces

from malib.utils.episode import Episode
from malib.utils.tianshou_batch import Batch
from malib.models.torch import make_net


class COMADiscreteCritic(nn.Module):
    def __init__(
        self,
        centralized_obs_space: gym.Space,
        action_space: gym.Space,
        net_type: str = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        super(COMADiscreteCritic, self).__init__()
        self.net_type = net_type
        self.net = make_net(
            observation_space=centralized_obs_space,
            action_space=action_space,
            device=device,
            net_type=net_type,
            **kwargs
        )

    def _build_inputs(self, agent_batch: Dict[str, Batch]) -> torch.Tensor:
        # concat states
        agents = list(agent_batch.keys())
        n_agents = len(agents)

        # concat by agent-axes: (batch, time_step(optional), inner_dim) -> (batch, time_step(optional), num_agent, inner_dim)
        states = torch.stack(
            [agent_batch[k][Episode.CUR_STATE] for k in agents], dim=-2
        )
        observations = torch.stack(
            [agent_batch[k][Episode.CUR_OBS] for k in agents], dim=-2
        )
        use_timestep = len(states.shape) > 3

        actions = torch.stack([agent_batch[k][Episode.ACTION] for k in agents], dim=-2)
        agent_mask = 1 - torch.eye(n_agents, device=states.device)
        # shape trans: (n_agents, n_agents) -> (n_agents^2, 1) -> (n_agents^2, n_action)
        # -> (n_agents, n_action * n_agents)
        agent_mask = (
            agent_mask.view(-1, 1).repeat(1, actions.shape[-1]).view(n_agents, -1)
        )

        if use_timestep:
            batch_size, time_step, _, _ = states.size()
            actions = actions.view(batch_size, time_step, 1, -1)
            actions = actions.repeat(1, 1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0).unsqueeze(0)
        else:
            (
                batch_size,
                _,
                _,
            ) = states.size()
            actions = actions.view(batch_size, 1, -1)
            actions = actions.repeat(1, n_agents, 1)
            agent_mask = agent_mask.unsqueeze(0)
        actions = actions * agent_mask

        # shape as: (batch, time_step(optional), agent, inner_dim)
        inputs = torch.cat([states, observations, actions], dim=-1)
        return inputs

    def forward(
        self, inputs: Union[Dict[str, Batch], torch.Tensor]
    ) -> Union[Tuple[torch.Tensor, Any], torch.Tensor]:
        if isinstance(inputs, Dict):
            inputs = self._build_inputs(inputs)
        assert isinstance(inputs, torch.Tensor), type(inputs)
        ori_shape = inputs.shape
        if self.net_type == "rnn":
            logits, hidden_state = self.net(inputs.view(-1, ori_shape[-1]))
            logits = logits.reshape(ori_shape[:-1] + (-1,))
            hidden_state = {
                k: v.reshape(ori_shape[:-1] + v.shape[1:])
                for k, v in hidden_state.items()
            }
            return (logits, hidden_state)
        else:
            return self.net(inputs)
