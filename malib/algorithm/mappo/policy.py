# -*- coding: utf-8 -*-
import copy
from malib.algorithm.mappo.actor_critic import RNNNet
from malib.algorithm.mappo.utils import PopArt, init_fc_weights
import os
import pickle
from typing import Tuple, Any, Dict
import gym
import torch
from torch import nn
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType
from malib.algorithm.common.misc import hard_update


class MAPPO(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs
    ):
        super(MAPPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.opt_cnt = 0
        self.register_state(self.opt_cnt, "opt_cnt")

        self._use_q_head = custom_config["use_q_head"]
        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )

        global_observation_space = custom_config["global_state_space"][
            kwargs["env_agent_id"]
        ]

        actor = RNNNet(
            self.model_config["actor"],
            observation_space,
            action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        critic = RNNNet(
            self.model_config["critic"],
            global_observation_space,
            action_space if self._use_q_head else gym.spaces.Discrete(1),
            self.custom_config,
            self.model_config["initialization"],
        )

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)

        if custom_config["use_popart"]:
            self.value_normalizer = PopArt(
                1, device=self.device, beta=custom_config["popart_beta"]
            )
            self.register_state(self.value_normalizer, "value_normalizer")

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy._actor = self_copy._actor.to(device)
        self_copy._critic = self_copy._critic.to(device)
        if self.custom_config["use_popart"]:
            self_copy.value_normalizer = self_copy.value_normalizer.to(device)
            self_copy.value_normalizer.tpdv = dict(dtype=torch.float32, device=device)
        return self_copy

    def compute_actions(self, observation, **kwargs):
        raise RuntimeError("Shouldn't use it currently")

    def forward_actor(self, obs, actor_rnn_states, rnn_masks):
        logits, actor_rnn_states = self.actor(obs, actor_rnn_states, rnn_masks)
        return logits, actor_rnn_states

    def compute_action(self, observation, **kwargs):
        actor_rnn_states = kwargs.get("actor_rnn_states", None)
        rnn_masks = kwargs.get("rnn_masks", None)
        logits, actor_rnn_states = self.actor(observation, actor_rnn_states, rnn_masks)
        if "action_mask" in kwargs:
            illegal_action_mask = torch.FloatTensor(
                1 - observation[..., : logits.shape[-1]]
            ).to(logits.device)
            assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
                illegal_action_mask.max(),
                illegal_action_mask.min(),
            )
            logits = logits - 1e10 * illegal_action_mask
        dist = torch.distributions.Categorical(logits=logits)
        extra_info = {}
        action_prob = dist.probs.detach().cpu().numpy()  # num_action

        # extra_info["action_probs"] = action_prob
        action = dist.sample().cpu().numpy()
        if "share_obs" in kwargs and kwargs["share_obs"] is not None:
            critic_rnn_states = kwargs.get("critic_rnn_states", None)
            value, critic_rnn_states = self.critic(
                kwargs["share_obs"], critic_rnn_states, rnn_masks
            )
            extra_info["value"] = value.detach().cpu().numpy()
            extra_info["critic_rnn_states"] = critic_rnn_states.detach().cpu().numpy()

        extra_info["actor_rnn_states"] = actor_rnn_states.detach().cpu().numpy()
        # XXX(ziyu): it seems that probs have some tiny numerical error, just use logits
        # return action, logits.detach().cpu().numpy(), extra_info
        return action, action_prob, extra_info

    def train(self):
        pass

    def eval(self):
        pass

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def dump(self, dump_dir):
        torch.save(self._actor, os.path.join(dump_dir, "actor.pt"))
        torch.save(self._critic, os.path.join(dump_dir, "critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        res = MAPPO(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs
        )

        actor = torch.load(os.path.join(dump_dir, "actor.pt"), res.device)
        critic = torch.load(os.path.join(dump_dir, "critic.pt"), res.device)

        hard_update(res._actor, actor)
        hard_update(res._critic, critic)
        return res
        
    # XXX(ziyu): test for this policy
    def state_dict(self):
        """Return state dict in real time"""

        res = {
            k: copy.deepcopy(v).cpu().state_dict()
            if isinstance(v, nn.Module)
            else v.state_dict()
            for k, v in self._state_handler_dict.items()
        }
        return res


if __name__ == "__main__":
    from malib.envs.gr_football import env, default_config
    import yaml

    cfg = yaml.load(open("mappo_grfootball/mappo_5_vs_5.yaml"))
    env = env(**default_config)
    custom_cfg = cfg["algorithms"]["MAPPO"]["custom_config"]
    custom_cfg.update({"global_state_space": env.observation_spaces})
    policy = MAPPO(
        "MAPPO",
        env.observation_spaces["team_0"],
        env.action_spaces["team_0"],
        cfg["algorithms"]["MAPPO"]["model_config"],
        custom_cfg,
        env_agent_id="team_0",
    )
    os.makedirs("play")
    policy.dump("play")
    MAPPO.load("play", env_agent_id="team_0")
