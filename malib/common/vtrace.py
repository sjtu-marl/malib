"""
An implementation of V-Trace algorithm, to taclking the imbalance of data use for on-policy training.
"""

from functools import reduce
from collections import namedtuple

import torch

from torch.nn import functional as F

from malib.utils.data import to_torch


VTraceRet = namedtuple("VTraceReturn", "vs,adv")
VTraceFromLogitsReturns = namedtuple(
    "VTraceFromLogitsReturns",
    ["vs", "adv", "log_rhos", "behaviour_log_prob", "target_log_prob"],
)


def _acc_func(acc, item):
    discount_t, c_t, delta_t = item
    return delta_t + discount_t * c_t * acc


def from_importance_weights(
    log_rhos: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    clip_rho: float,
    clip_pg_rho: float,
):
    """Calculates V-trace values from log importance weights.

    Args:
        log_rhos (torch.Tensor): A float tensor of shape [T, B, N_ACTIONS] representing the log importantce sampling weights, i.e., log[target_p(a) / behavior_p(a)].
        discounts (torch.Tensor): A float tensor of shape [T, B] with discounts encountered by following the behaviour policy.
        rewards (torch.Tensor): A float tensor of shape [T, B] containing rewards generated by following the behavior policy.
        values (torch.Tensor): A float tensor of shape [T, B] with the value function estimates wrt. the target policy.
        bootstrap_values (torch.Tensor): A float of shape [B] with teh value function estimate at time T.
        clip_rho (float): A float scalar with the clipping threshold for importance weights (rho) when calculating the baseline targets (Vs).
        clip_pg_rho (float): A float scalar with the clipping threshold on rho_s in rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    """

    log_rhos = to_torch(log_rhos, dtype=torch.float32)
    discounts = to_torch(discounts, dtype=torch.float32)
    rewards = to_torch(rewards, dtype=torch.float32)
    values = to_torch(values, dtype=torch.float32)
    bootstrap_values = to_torch(bootstrap_values, dtype=torch.float32)

    # shape assert
    assert log_rhos.shape.__len__() == 3, log_rhos.shape
    assert values.shape.__len__() == 2, values.shape
    assert bootstrap_values.shape.__len__() == 1, bootstrap_values.shape
    assert rewards.shape.__len__() == 2, rewards.shape
    assert discounts.shape.__len__() == 2, discounts.shape

    rhos = torch.exp(log_rhos)
    clipped_rho = torch.minimum(clip_rho, rhos) if clip_rho else rhos
    cs = torch.minimum(1.0, rhos)
    values_t_plus_1 = torch.concat([values[1:], bootstrap_values.unsqueeze(0)], dim=0)
    deltas = clipped_rho * (rewards + discounts * values_t_plus_1 - values)

    sequences = (discounts, cs, deltas)
    vs_minus_v_xs = reduce(_acc_func, sequences)

    assert vs_minus_v_xs.shape == values, (vs_minus_v_xs.shape, values.shape)
    vs = vs_minus_v_xs + values

    # compute advantages
    vs_t_plus_1 = torch.concat([vs[1:], bootstrap_values.unsqueeze(0)], axis=0)
    clippped_pg_rho = torch.minimum(clip_pg_rho, rhos) if clip_pg_rho else rhos
    advantages = clippped_pg_rho * (rewards + discounts * vs_t_plus_1 - values)

    return VTraceRet(vs=vs.detach(), adv=advantages.detach())


def vtrace(
    behavior_logits: torch.Tensor,
    target_logits: torch.Tensor,
    actions: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    clip_rho: float = 1.0,
    clip_pg_rho: float = 1.0,
):
    """Calculates V-trace actor critic targets for softmax policies as introduced in

    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" by Espeholt, Soyer, Munos et al.

    Args:
        behavior_logits (torch.Tensor): A float tensor of shape [T, B, N_ACTIONS] with un-normalized log-probabilities parameterizing the softmax behaviour policy.
        target_logits (torch.Tensor): A float tensor of shape [T, B, N_ACTIONS] with un-normalized log-probabilities parameterizing the softmax target policy.
        actions (torch.Tensor): An int tensor of shape [T, B] of actions sampled from the behavior policy.
        discounts (torch.Tensor): A float tensor of shape [T, B] with the discount encountered when following the behavior policy.
        rewards (torch.Tensor): A float tensor of shape [T, B] with the rewards generated by following the behavior policy.
        values (torch.Tensor): A float tensor of shape [T, B] with the value function estimates wrt. the target policy.
        bootstrap_values (torch.Tensor): A float of shape [B] with the value function estimate at time T.
        clip_rho (float, optional): A float scalar with the clipping threshold for importance weights (RHO) when calculating the baseline targets (Vs), i.e., \bar{rho} in the paper. Defaults to 1..
        clip_pg_rho (float, optional): A float scalar with the clipping threshold on rho_s in rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). Defaults to 1.
    """

    behavior_logits = to_torch(behavior_logits, dtype=torch.float32)
    target_logits = to_torch(target_logits, dtype=torch.float32)
    actions = to_torch(actions, dtype=torch.int32)

    # shape checking
    assert behavior_logits.shape.__len__() == 3, behavior_logits.shape
    assert target_logits.shape.__len__() == 3, target_logits.shape
    assert actions.shape.__len__() == 2, actions.shape

    # compute log probs for behavior and target policy
    behavior_log_prob = F.cross_entropy(input=behavior_logits, target=actions)
    target_log_prob = F.cross_entropy(input=target_logits, target=actions)

    log_rhos = target_log_prob - behavior_log_prob
    vtrace_ret = from_importance_weights(
        log_rhos, discounts, rewards, values, bootstrap_values, clip_rho, clip_pg_rho
    )

    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behaviour_log_prob=behavior_log_prob,
        target_log_prob=target_log_prob ** vtrace_ret._asdict(),
    )