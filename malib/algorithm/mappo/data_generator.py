from collections import defaultdict
from typing import Dict
import torch
import numpy as np
from malib.algorithm.mappo.vtrace import compute_vtrace
from malib.utils.episode import Episode


def get_part_data_from_batch(batch_data, idx):
    # (ziyu): suppose the first dimension is batch_size
    res = {}
    for k, v in batch_data.items():
        res[k] = v[idx]
    return res


def compute_return(policy, batch, mode="gae"):
    cm_cfg = policy.custom_config
    gamma, gae_lambda = cm_cfg["gamma"], cm_cfg["gae"]["gae_lambda"]
    values, rewards, dones = (
        # FIXME(ziyu): for debugging
        np.zeros_like(batch[Episode.REWARD]),
        # batch[Episode.STATE_VALUE],
        batch[Episode.REWARD],
        batch[Episode.DONE],
    )
    if cm_cfg["use_popart"]:
        values = policy.value_normalizer.denormalize(values)

    if mode == "gae":
        return compute_gae(values, rewards, dones, gamma, gae_lambda)
    elif mode == "vtrace":
        return compute_vtrace(
            policy,
            batch[Episode.CUR_OBS],
            rewards,
            values,
            dones,
            batch["rnn_state_0"],
            batch[Episode.ACTION],
            batch[Episode.ACTION_DIST],
            gamma,
            cm_cfg["vtrace"]["clip_rho_threshold"],
            cm_cfg["vtrace"]["clip_pg_rho_threshold"],
        )
    else:
        raise ValueError("Unexpected return mode: {}".format(mode))


def compute_gae(value, reward, done, gamma, gae_lambda):
    assert len(reward.shape) == 4, (reward.shape, done.shape, value.shape)
    B, Tp1, N, _ = reward.shape
    assert list(value.shape) == [B, Tp1, N, 1] and list(done.shape) == [B, Tp1, N, 1]
    value = np.transpose(value, (1, 0, 2, 3))
    done = np.transpose(done, (1, 0, 2, 3))
    reward = np.transpose(reward, (1, 0, 2, 3))

    gae, ret = 0, np.zeros_like(reward)
    for t in reversed(range(Tp1 - 1)):
        delta = reward[t] + gamma * (1 - done[t]) * value[t + 1] - value[t]
        gae = delta + gamma * gae_lambda * (1 - done[t]) * gae
        ret[t] = gae + value[t]

    return {"return": ret.transpose((1, 0, 2, 3))}


def simple_data_generator(batch, num_mini_batch, device):
    # XXX(ziyu): if we put all data on GPUs, mini-batch cannot work when we don't have enough GPU memory
    batch_size, _ = batch[Episode.CUR_OBS].shape

    mini_batch_size = batch_size // num_mini_batch

    assert mini_batch_size > 0

    rand = torch.randperm(batch_size).numpy()
    for i in range(0, batch_size, mini_batch_size):
        tmp_slice = slice(i, min(batch_size, i + mini_batch_size))
        tmp_batch = get_part_data_from_batch(batch, rand[tmp_slice])
        yield tmp_batch


def recurrent_generator(data, num_mini_batch, rnn_data_chunk_length, device):
    batch = {k: d.copy() for k, d in data.items()}
    # original shape is [fragment_length, batch_size, num_agent, ...]
    def _cast(x):
        return x.permute(1, 2, 0, 3).reshape(-1, *x.shape[3:])

    for k in batch:
        if isinstance(batch[k], np.ndarray):
            batch[k] = torch.FloatTensor(batch[k])  # .to(device)
            # FIXME（ziyu): the put on GPU operation here should be considered in detail
        if k not in ["rnn_state_0", "rnn_state_1"]:
            batch[k] = _cast(batch[k])
        else:
            batch[k] = batch[k].permute(1, 2, 0, 3, 4).reshape(-1, *batch[k].shape[3:])
    batch_size, _ = batch[Episode.CUR_OBS].shape

    data_chunks = batch_size // rnn_data_chunk_length  # [C=r*T*M/L]
    mini_batch_size = data_chunks // num_mini_batch

    rand = torch.randperm(data_chunks).numpy()
    sampler = [
        rand[i * mini_batch_size : (i + 1) * mini_batch_size]
        for i in range(num_mini_batch)
    ]
    for indices in sampler:
        tmp_batch_list = defaultdict(list)
        for index in indices:
            ind = index * rnn_data_chunk_length
            for k in batch:
                if k not in ["rnn_state_0", "rnn_state_1"]:
                    tmp_batch_list[k].append(
                        batch[k][ind : ind + rnn_data_chunk_length]
                    )
                else:
                    tmp_batch_list[k].append(batch[k][ind])

        T, N = rnn_data_chunk_length, mini_batch_size

        tmp_batch = {}
        for k in batch:
            if k not in ["rnn_state_0", "rnn_state_1"]:
                tmp_batch[k] = torch.stack(tmp_batch_list[k], dim=1)
                tmp_batch[k] = tmp_batch[k].reshape(N * T, *tmp_batch[k].shape[2:])
            else:
                tmp_batch[k] = torch.stack(tmp_batch_list[k])

        yield {k: v.to(device) for k, v in tmp_batch.items()}
