# -*- coding: utf-8 -*-
from collections import defaultdict
from logging import Logger
import time
from typing import Dict, List
import numpy as np
import ray
import torch
from malib.algorithm.mappo.data_generator import recurrent_generator
from malib.algorithm.mappo.loss import MAPPOLoss

from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.gr_football.env import GRFEpisodeInfo
from malib.utils.metrics import get_metric


def _parse_episode_infos(episode_infos: List[GRFEpisodeInfo]) -> Dict[str, List]:
    # FIXME(ziyu): now reward is actually a num_agent * 1 array for each
    res = defaultdict(list)
    for episode_info in episode_infos:
        for aid, epi in episode_info.items():
            for k, v in epi._record.items():
                tag = f"metric/{k}/{aid}"
                res[tag].append(v)
    return res


def grf_simultaneous(
    env,
    num_envs,
    fragment_length,
    max_step,  # XXX(ziyu): deprecated param
    behavior_policies,
    buffer_desc,
    send_interval=50,
    dataset_server=None,
):
    if buffer_desc is not None:
        agent_buffers = {agent: defaultdict(list) for agent in buffer_desc.agent_id}
    else:
        agent_buffers = None

    rets = env.reset(
        limits=num_envs,
        fragment_length=fragment_length,
        env_reset_kwargs={"max_step": max_step},
    )

    # observations = {
    #     aid: agent_interfaces[aid].transform_observation(obs)
    #     for aid, obs in rets[Episode.CUR_OBS].items()
    # } # FIXME(ziyu): should add transform_obs
    observations = rets[Episode.CUR_OBS]
    states = rets[Episode.CUR_STATE]

    dones, critic_rnn_states, actor_rnn_states = {}, {}, {}
    for agent_id in observations:
        policy = behavior_policies[agent_id]
        n_rollout_threads, n_agent = observations[agent_id].shape[:2]
        critic_rnn_states[agent_id] = np.zeros(
            (
                n_rollout_threads,
                n_agent,
                policy.custom_config["rnn_layer_num"],
                policy.model_config["critic"]["layers"][-1]["units"],
            )
        )
        actor_rnn_states[agent_id] = np.zeros(
            (
                n_rollout_threads,
                n_agent,
                policy.custom_config["rnn_layer_num"],
                policy.model_config["actor"]["layers"][-1]["units"],
            )
        )
        dones[agent_id] = np.zeros((n_rollout_threads, n_agent, 1))

    split_cast = lambda x: np.array(np.split(x, n_rollout_threads))

    while not env.is_terminated():
        actions, action_dists, action_masks, avail_actions = {}, {}, {}, {}
        values, extra_infos = {}, {}
        # XXX(ziyu): the procedure can be generalized as a rollout function with
        #  GAE lambda computation process, and then make the GAE as another code block

        for agent in observations:
            avail_actions[agent] = observations[agent][..., :19]
            action, action_dist, extra_info = behavior_policies[agent].compute_action(
                # (ziyu): use concatenate to make [num_env, num_agent, ...] -> [-1, ...]
                np.concatenate(observations[agent]),
                action_mask=np.concatenate(avail_actions[agent]),
                share_obs=np.concatenate(states[agent]),
                actor_rnn_states=np.concatenate(actor_rnn_states[agent]),
                critic_rnn_states=np.concatenate(critic_rnn_states[agent]),
                rnn_masks=np.concatenate(dones[agent]),
            )

            actions[agent] = split_cast(action)
            action_dists[agent] = split_cast(action_dist)
            # import pdb; pdb.set_trace()
            # assert np.equal(action_dists[agent].sum(-1), 1).all()
            values[agent] = split_cast(extra_info["value"])
            extra_infos[agent] = extra_info
        rets = env.step(actions)
        dones, rewards, infos = (
            rets[Episode.DONE],
            rets[Episode.REWARD],
            rets[Episode.INFO],
        )

        if dataset_server:
            for agent in env.trainable_agents:
                shape0 = observations[agent].shape[:2]
                data_to_insert = {
                    Episode.CUR_OBS: observations[agent],
                    Episode.ACTION: actions[agent][..., None],
                    Episode.ACTION_DIST: action_dists[agent],
                    Episode.REWARD: rewards[agent],
                    Episode.DONE: dones[agent],
                    "active_mask": np.ones((*shape0, 1)),
                    "available_action": observations[agent][..., :19],
                    # TODO(ziyu): use action mask by parse obs.
                    "value": values[agent],
                    # "return": np.zeros((*shape0, 1)),
                    "share_obs": states[agent],
                    "actor_rnn_states": actor_rnn_states[agent],
                    "critic_rnn_states": critic_rnn_states[agent],
                }
                for k, v in data_to_insert.items():
                    agent_buffers[agent][k].append(v)

                actor_rnn_states[agent] = split_cast(
                    extra_infos[agent]["actor_rnn_states"]
                )
                critic_rnn_states[agent] = split_cast(
                    extra_infos[agent]["critic_rnn_states"]
                )

        # observations = {
        #     aid: agent_interfaces[aid].transform_observation(obs)
        #     for aid, obs in rets[Episode.CUR_OBS].items()
        # }
        observations = rets[Episode.CUR_OBS]
        states = rets[Episode.CUR_STATE]

        done = all([d.all() for d in dones.values()])

    bootstrap_values = {aid: np.zeros_like(values[aid]) for aid in observations}
    if not done and buffer_desc is not None:
        for agent in agent_buffers:
            _, _, extra_info = behavior_policies[agent].compute_action(
                np.concatenate(observations[agent]),
                policy_id=behavior_policies[agent],
                share_obs=np.concatenate(states[agent]),
                actor_rnn_states=np.concatenate(actor_rnn_states[agent]),
                critic_rnn_states=np.concatenate(critic_rnn_states[agent]),
                rnn_masks=np.concatenate(dones[agent]),
            )
            actor_rnn_states[agent] = split_cast(
                extra_infos[agent]["actor_rnn_states"]
            )
            critic_rnn_states[agent] = split_cast(
                extra_infos[agent]["critic_rnn_states"]
            )
            bootstrap_values[agent] = split_cast(extra_info["value"])

    if dataset_server is not None:
        for agent in agent_buffers:
            boostrap_data = {
                    Episode.CUR_OBS: observations[agent] * 0,
                    Episode.ACTION: actions[agent][..., None] * 0,
                    Episode.ACTION_DIST: action_dists[agent] * 0,
                    Episode.REWARD: rewards[agent] * 0,
                    Episode.DONE: dones[agent] * 0,
                    "active_mask": np.empty((*shape0, 1)),
                    "available_action": observations[agent][..., :19] * 0,
                    # TODO(ziyu): use action mask by parse obs.
                    "value": bootstrap_values[agent],
                    # "return": np.zeros((*shape0, 1)),
                    "share_obs": states[agent],
                    "actor_rnn_states": actor_rnn_states[agent] * 0,
                    "critic_rnn_states": critic_rnn_states[agent] * 0,
                }
            for k, v in boostrap_data.items():
                agent_buffers[agent][k].append(v)
            for k, v_list in agent_buffers[agent].items():
                agent_buffers[agent][k] = np.stack(v_list)

        # compute_gae(bootstrap_values)

        for agent in agent_buffers:
            for k, v in agent_buffers[agent].items():
                dims = len(v.shape)
                residual_dims = range(2, dims)
                agent_buffers[agent][k] = np.transpose(v, (1, 0, *residual_dims))


        indices = None
        buffer_desc.batch_size = num_envs
        while indices is None:
            batch = ray.get(dataset_server.get_producer_index.remote(buffer_desc))
            indices = batch.data

        # shuffle_idx = np.random.permutation(len(indices))

        buffer_desc.data = agent_buffers
        buffer_desc.indices = indices
        assert len(indices) == num_envs, (len(indices), num_envs)
        dataset_server.save.remote(buffer_desc)

    results = _parse_episode_infos(env.episode_infos)
    return results, env.batched_step_cnt * len(env.possible_agents)
