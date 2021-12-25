import torch
from torch.distributions.categorical import Categorical
import numpy as np


def target_logp_from_policy_and_act(policy, obs, act, rnn_states, masks):
    if obs.ndim == 4:  # [B, T, N, *]
        B, T, N, _ = obs.shape
        _cast = lambda x: torch.tensor(
            x.reshape((B * T * N, -1)), dtype=torch.float32, device=policy.device
        )
        obs = _cast(obs)
        masks = _cast(masks)
        rnn_states = torch.tensor(
            rnn_states.reshape((B * T * N, *rnn_states.shape[-2:])),
            dtype=torch.float32,
            device=policy.device,
        )
    logits, _ = policy.actor(obs, rnn_states, masks)
    logits = logits.reshape((B, T, N, -1)).detach()
    dist = Categorical(logits=logits)
    act = torch.tensor(act.squeeze(-1), dtype=torch.float32, device=policy.device)
    return dist.log_prob(act).unsqueeze(-1).cpu().numpy()


def behavior_logp_from_prob_and_act(act_dist, act):
    dist = Categorical(logits=torch.FloatTensor(act_dist))
    act = torch.LongTensor(act.squeeze(-1))
    return dist.log_prob(act).unsqueeze(-1).numpy()


def compute_vtrace(
    policy,
    obs,
    reward,
    value,
    done,
    rnn_state,
    act,
    act_dist,
    gamma,
    clip_rho_threshold,
    clip_pg_rho_threshold,
):
    behavior_logp = behavior_logp_from_prob_and_act(act_dist, act)
    target_logp = target_logp_from_policy_and_act(policy, obs, act, rnn_state, done)
    log_rhos = target_logp - behavior_logp
    log_rhos, values = log_rhos[:, :-1], value[:, :-1]
    rewards, dones = reward[:, :-1], done[:, :-1]

    bootstrap_values = values[:, -1]

    vs, pg_adv = vtrace_return(
        log_rhos,
        gamma,
        rewards,
        values,
        bootstrap_values,
        dones,
        clip_rho_threshold,
        clip_pg_rho_threshold,
    )

    v_hub, ret_hub = np.zeros_like(value), np.zeros_like(value)
    v_hub[:, :-1], ret_hub[:, :-1] = vs, (vs + pg_adv)

    return {"value": v_hub, "return": ret_hub}


def vtrace_return(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_values,
    dones,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    discounts = np.array(discounts, dtype=np.float32)
    clip_rho_threshold = np.array(clip_rho_threshold, dtype=np.float32)
    clip_pg_rho_threshold = np.array(clip_pg_rho_threshold, dtype=np.float32)

    rho_rank = log_rhos.ndim
    B, T, N, _ = values.shape
    assert values.ndim == rho_rank and rewards.ndim == rho_rank, (
        log_rhos.shape,
        values.shape,
        rewards.shape,
    )
    assert bootstrap_values.ndim == rho_rank - 1

    log_rhos = log_rhos.transpose((1, 0, 2, 3))
    values = values.transpose((1, 0, 2, 3))
    rewards = rewards.transpose((1, 0, 2, 3))
    dones = dones.transpose((1, 0, 2, 3))

    rhos = np.exp(log_rhos)
    clipped_rhos = np.minimum(clip_rho_threshold, rhos)
    cs = np.minimum(1.0, rhos)

    values_tp1 = np.concatenate(
        [values[1:], np.expand_dims(bootstrap_values, 0)], axis=0
    )

    deltas = clipped_rhos * (rewards + (1 - dones) * discounts * values_tp1 - values)

    itermediate_v = np.zeros_like(bootstrap_values)
    vs_minus_v_xs = np.zeros_like(values)
    for t in reversed(range(T)):
        itermediate_v = deltas[t] + (1 - dones[t]) * discounts * cs[t] * itermediate_v
        vs_minus_v_xs[t] = itermediate_v

    vs = vs_minus_v_xs + values

    vs_tp1 = np.concatenate([vs[1:], np.expand_dims(bootstrap_values, 0)], axis=0)

    clipped_pg_rhos = np.minimum(clip_pg_rho_threshold, rhos)

    pg_advantages = clipped_pg_rhos * (
        rewards + (1 - dones) * discounts * vs_tp1 - values
    )

    return vs.transpose((1, 0, 2, 3)), pg_advantages.transpose((1, 0, 2, 3))
