from collections import defaultdict
import torch
import numpy as np
from malib.backend.datapool.offline_dataset_server import Episode


def get_part_data_from_batch(batch_data, idx):
    # (ziyu): suppose the first dimension is batch_size
    res = {}
    for k, v in batch_data.items():
        res[k] = v[idx]
    return res


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
        if k not in ["actor_rnn_states", "critic_rnn_states"]:
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
                if k not in ["actor_rnn_states", "critic_rnn_states"]:
                    tmp_batch_list[k].append(
                        batch[k][ind : ind + rnn_data_chunk_length]
                    )
                else:
                    tmp_batch_list[k].append(batch[k][ind])

        T, N = rnn_data_chunk_length, mini_batch_size

        tmp_batch = {}
        for k in batch:
            if k not in ["actor_rnn_states", "critic_rnn_states"]:
                tmp_batch[k] = torch.stack(tmp_batch_list[k], dim=1)
                tmp_batch[k] = tmp_batch[k].reshape(N * T, *tmp_batch[k].shape[2:])
            else:
                tmp_batch[k] = torch.stack(tmp_batch_list[k])

        yield {k: v.to(device) for k, v in tmp_batch.items()}
