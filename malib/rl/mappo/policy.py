from typing import Tuple, Any, Dict, List

import copy
import os
import pickle
import wrapt
import tree

import gym
import torch
import numpy as np

from torch import nn

from malib.utils.typing import DataTransferType
from malib.utils.episode import Episode
from malib.rl.common.policy import Policy
from malib.rl.common.misc import hard_update
from malib.rl.mappo.actor_critic import RNNNet
from malib.rl.mappo.utils import PopArt, init_fc_weights


@wrapt.decorator
def shape_adjusting(wrapped, instance, args, kwargs):
    """
    A wrapper that adjust the inputs to corrent shape.
    e.g.
        given inputs with shape (n_rollout_threads, n_agent, ...)
        reshape it to (n_rollout_threads * n_agent, ...)
    """
    offset = len(instance.preprocessor.shape)
    original_shape_pre = kwargs[Episode.CUR_OBS].shape[:-offset]
    num_shape_ahead = len(original_shape_pre)

    def adjust_fn(x):
        if isinstance(x, np.ndarray):
            return np.reshape(x, (-1,) + x.shape[num_shape_ahead:])
        else:
            return x

    def recover_fn(x):
        try:
            if isinstance(x, np.ndarray):
                return np.reshape(x, original_shape_pre + x.shape[1:])
            else:
                return x
        except ValueError as e:
            raise e

    adjusted_args = tree.map_structure(adjust_fn, args)
    adjusted_kwargs = tree.map_structure(adjust_fn, kwargs)

    # print("adjstyee:", {k: v.shape if isinstance(v, np.ndarray) else 0 for k, v in adjusted_kwargs.items()})

    rets = wrapped(*adjusted_args, **adjusted_kwargs)

    recover_rets = tree.map_structure(recover_fn, rets)

    return recover_rets


class MAPPO(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
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
        self.env_agent_id = kwargs["env_agent_id"]

        # TODO(ming): will collect to custom config
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

    def get_initial_state(self, batch_size) -> List[DataTransferType]:
        return [
            np.zeros((batch_size, rnn_net.rnn_layer_num, rnn_net.rnn_state_size))
            for rnn_net in [self._actor, self._critic]
        ]

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

    @shape_adjusting
    def compute_action(self, observation, **kwargs):
        actor_rnn_states, critic_rnn_states = kwargs[Episode.RNN_STATE]

        rnn_masks = kwargs.get(Episode.DONE, False)
        logits, actor_rnn_states = self.actor(
            observation.copy(), actor_rnn_states.copy(), rnn_masks
        )
        actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
        if Episode.ACTION_MASK in kwargs:
            illegal_action_mask = torch.FloatTensor(1 - kwargs[Episode.ACTION_MASK]).to(
                logits.device
            )
            assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
                illegal_action_mask.max(),
                illegal_action_mask.min(),
            )
            logits = logits - 1e10 * illegal_action_mask
        dist = torch.distributions.Categorical(logits=logits)
        action_prob = dist.probs.detach().cpu().numpy()  # num_action

        # extra_info["action_probs"] = action_prob
        action = dist.sample().cpu().numpy()
        if Episode.CUR_STATE in kwargs and kwargs[Episode.CUR_STATE] is not None:
            value, critic_rnn_states = self.critic(
                kwargs[Episode.CUR_STATE].copy(), critic_rnn_states.copy(), rnn_masks
            )
            critic_rnn_states = critic_rnn_states.detach().cpu().numpy()

        return action, action_prob, [actor_rnn_states, critic_rnn_states]

    @shape_adjusting
    def value_function(self, *args, **kwargs):
        # FIXME(ziyu): adjust shapes
        state = kwargs[Episode.CUR_STATE]
        rnn_state_key_for_train = f"{Episode.RNN_STATE}_1"
        if rnn_state_key_for_train in kwargs:
            critic_rnn_state = kwargs[rnn_state_key_for_train]
        else:
            critic_rnn_state = kwargs[Episode.RNN_STATE][1]
        rnn_mask = kwargs.get(Episode.DONE, False)
        with torch.no_grad():
            value, _ = self.critic(state.copy(), critic_rnn_state.copy(), rnn_mask)
        return value.cpu().numpy()

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
            **kwargs,
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
