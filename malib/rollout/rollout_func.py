"""
Users can register their rollout func here, with the same parameters list like method `sequential`
and return a Dict-like metric results.

Examples:
    >>> def custom_rollout_function(
    ...     agent_interfaces: List[env.AgentInterface],
    ...     env_desc: Dict[str, Any],
    ...     metric_type: str,
    ...     max_iter: int,
    ...     behavior_policy_mapping: Dict[AgentID, PolicyID],
    ... ) -> Dict[str, Any]

In your custom rollout function, you can decide extra data
you wanna save by specifying extra columns when Episode initialization.
"""
import time
from collections import defaultdict

import numpy as np

from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.metrics import get_metric
from malib.utils.typing import AgentID, Dict
from malib.utils.preprocessor import get_preprocessor


def sequential(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
):
    """ Rollout in sequential manner """
    res1, res2 = [], []
    env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))
    # for _ in range(num_episode):
    env.reset()

    # metric.add_episode(f"simulation_{policy_combination_mapping}")
    metric = get_metric(metric_type)(env.possible_agents)
    if behavior_policy_mapping is None:
        for agent in agent_interfaces.values():
            agent.reset()
    behavior_policy_mapping = behavior_policy_mapping or {
        _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
    }
    agent_episode = {
        agent: Episode(
            env_desc["id"], behavior_policy_mapping[agent], capacity=max_iter
        )
        for agent in (trainable_pairs or env.possible_agents)
    }

    (observations, actions, action_dists, next_observations, rewards, dones, infos,) = (
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
    )

    for aid in env.agent_iter(max_iter=max_iter):
        observation, reward, done, info = env.last()

        if isinstance(observation, dict):
            info = {"action_mask": observation["action_mask"]}
            action_mask = observation["action_mask"]
        else:
            action_mask = np.ones(
                get_preprocessor(env.action_spaces[aid])(env.action_spaces[aid]).size
            )
        observation = agent_interfaces[aid].transform_observation(
            observation, behavior_policy_mapping[aid]
        )
        observations[aid].append(observation)
        rewards[aid].append(reward)
        dones[aid].append(done)
        info["policy_id"] = behavior_policy_mapping[aid]

        if not done:
            action, action_dist, extra_info = agent_interfaces[aid].compute_action(
                observation, **info
            )
            actions[aid].append(action)
            action_dists[aid].append(action_dist)
        else:
            info["policy_id"] = behavior_policy_mapping[aid]
            action = None
        env.step(action)
        metric.step(
            aid,
            behavior_policy_mapping[aid],
            observation=observation,
            action=action,
            action_dist=action_dist,
            reward=reward,
            done=done,
            info=info,
        )

    # metric.end()

    for k in agent_episode:
        obs = observations[k]
        cur_len = len(obs)
        agent_episode[k].fill(
            **{
                Episode.CUR_OBS: np.stack(obs[: cur_len - 1]),
                Episode.NEXT_OBS: np.stack(obs[1:cur_len]),
                Episode.DONES: np.stack(dones[k][1:cur_len]),
                Episode.REWARDS: np.stack(rewards[k][1:cur_len]),
                Episode.ACTIONS: np.stack(actions[k][: cur_len - 1]),
                Episode.ACTION_DIST: np.stack(action_dists[k][: cur_len - 1]),
            }
        )
    return (
        metric.parse(
            agent_filter=tuple(trainable_pairs.keys())
            if trainable_pairs is not None
            else None
        ),
        agent_episode,
    )


def simultaneous(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
):
    """Do not support next action mask.

    :param trainable_pairs:
    :param agent_interfaces:
    :param env_desc:
    :param metric_type:
    :param max_iter:
    :param behavior_policy_mapping:
    :return:
    """

    env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))

    metric = get_metric(metric_type)(
        env.possible_agents if trainable_pairs is None else list(trainable_pairs.keys())
    )

    if behavior_policy_mapping is None:
        for agent in agent_interfaces.values():
            agent.reset()

    behavior_policy_mapping = behavior_policy_mapping or {
        _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
    }

    agent_episode = {
        agent: Episode(
            env_desc["id"],
            behavior_policy_mapping[agent],
            capacity=max_iter,
            other_columns=["times"],
        )
        for agent in (trainable_pairs or env.possible_agents)
    }

    done = False
    step = 0
    observations = env.reset()

    for agent, obs in observations.items():
        observations[agent] = agent_interfaces[agent].transform_observation(
            obs, policy_id=behavior_policy_mapping[agent]
        )

    while step < max_iter and not done:
        actions, action_dists = {}, {}
        for agent, interface in agent_interfaces.items():
            action, action_dist, extra_info = agent_interfaces[agent].compute_action(
                observations[agent], policy_id=behavior_policy_mapping[agent]
            )
            actions[agent] = action
            action_dists[agent] = action_dist

        next_observations, rewards, dones, infos = env.step(actions)

        for agent, interface in agent_interfaces.items():
            obs = next_observations[agent]
            if obs is not None:
                next_observations[agent] = interface.transform_observation(
                    obs, policy_id=behavior_policy_mapping[agent]
                )
            else:
                next_observations[agent] = np.zeros_like(observations[agent])
        time_stamp = time.time()
        for agent in agent_episode:
            agent_episode[agent].insert(
                **{
                    Episode.CUR_OBS: np.expand_dims(observations[agent], 0),
                    Episode.ACTIONS: np.asarray([actions[agent]]),
                    Episode.REWARDS: np.asarray([rewards[agent]]),
                    Episode.ACTION_DIST: np.expand_dims(action_dists[agent], 0),
                    Episode.NEXT_OBS: np.expand_dims(next_observations[agent], 0),
                    Episode.DONES: np.asarray([dones[agent]]),
                    "times": np.asarray([time_stamp]),
                }
            )
            metric.step(
                agent,
                behavior_policy_mapping[agent],
                observation=observations[agent],
                action=actions[agent],
                reward=rewards[agent],
                action_dist=action_dists[agent],
                done=dones[agent],
            )
        step += 1
        done = any(dones.values())

    return (
        metric.parse(agent_filter=tuple(agent_episode)),
        agent_episode,
    )


def rollout_wrapper(
    agent_episodes: Dict[AgentID, Episode] = None, rollout_type="sequential"
):
    """Rollout wrapper accept a dict of episodes outside.

    :param Dict[AgentID,Episode] agent_episodes: A dict of agent episodes.
    :param str rollout_type: Specify rollout styles. Default to `sequential`, choices={sequential, simultaneous}.
    :return: A function
    """

    handler = sequential if rollout_type == "sequential" else simultaneous

    def func(
        trainable_pairs,
        agent_interfaces,
        env_desc,
        metric_type,
        max_iter,
        behavior_policy_mapping=None,
    ):
        statistic, episodes = handler(
            trainable_pairs,
            agent_interfaces,
            env_desc,
            metric_type,
            max_iter,
            behavior_policy_mapping=behavior_policy_mapping,
        )
        if agent_episodes is not None:
            for agent, episode in episodes.items():
                agent_episodes[agent].insert(**episode.data)
        return statistic, episodes

    return func


def multi_agent_rollout(multi_agent_episode):
    """Not READY"""

    def rollout_fn(
        trainable_pairs,
        agent_interfaces,
        env_desc,
        metric_type,
        max_iter,
        behavior_policy_mapping=None,
    ):
        env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))
        metric = get_metric(metric_type)(env.possible_agents)
        behavior_policy_mapping = behavior_policy_mapping or {
            _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
        }

        observations = defaultdict(list)
        states = defaultdict(list)
        actions = defaultdict(list)
        rewards = defaultdict(list)
        action_dists = defaultdict(list)
        next_observations = defaultdict(list)
        next_states = defaultdict(list)
        dones = defaultdict(list)
        next_action_masks = defaultdict(list)
        infos = defaultdict(list)

        obs_dict, state_dict, action_mask_dict = env.reset()

        cnt = 0
        while cnt < max_iter:
            act_dict, act_dist_dict = {}, {}
            for aid, obs in obs_dict.items():
                info = {}
                info["action_mask"] = action_mask_dict[aid]

                # XXX(ziyu): what is "info" we need? I think we may need to pass the true
                #  info_dict[aid] to compute_action
                info["policy_id"] = behavior_policy_mapping[aid]
                act_dict[aid], _, extra_info = agent_interfaces[aid].compute_action(
                    obs, **info
                )
                act_dist_dict[aid] = extra_info["action_probs"]
            (
                next_obs_dict,
                next_state_dict,
                rew_dict,
                done_dict,
                action_mask_dict,
                info_dict,
            ) = env.step(act_dict)

            for aid in env.agents:
                (
                    _obs,
                    _state,
                    _act,
                    _act_dist,
                    _rew,
                    _n_obs,
                    _n_state,
                    _done,
                    _action_mask,
                    _info,
                ) = (
                    obs_dict[aid],
                    state_dict[aid],
                    act_dict[aid],
                    act_dist_dict[aid],
                    rew_dict[aid],
                    next_obs_dict[aid],
                    next_state_dict[aid],
                    done_dict[aid],
                    action_mask_dict[aid],
                    info_dict[aid],
                )
                observations[aid].append(_obs)
                states[aid].append(_state)
                actions[aid].append(_act)
                action_dists[aid].append(_act_dist)
                next_observations[aid].append(_n_obs)
                next_states[aid].append(_n_state)
                rewards[aid].append(_rew)
                dones[aid].append(_done)
                next_action_masks[aid].append(_action_mask)
                infos[aid].append(_info)

                # metric.step(aid, _obs, _act, _rew, _done, _info)
                metric.step(aid, behavior_policy_mapping[aid], reward=_rew, info=_info)

            obs_dict = next_obs_dict
            state_dict = next_state_dict

            if all(done_dict.values()):
                break

        multi_agent_episode.insert(
            **{
                aid: {
                    Episode.CUR_OBS: np.stack(observations[aid]),
                    Episode.CUR_STATE: np.stack(states[aid]),
                    Episode.NEXT_OBS: np.stack(next_observations[aid]),
                    Episode.NEXT_STATE: np.stack(next_states[aid]),
                    Episode.DONES: np.stack(dones[aid]),
                    Episode.REWARDS: np.stack(rewards[aid]),
                    Episode.ACTIONS: np.stack(actions[aid]),
                    Episode.ACTION_DIST: np.stack(action_dists[aid]),
                    Episode.NEXT_ACTION_MASK: np.stack(next_action_masks[aid]),
                }
                for aid in observations
            }
        )

        return metric.parse(), {}

    return rollout_fn


def get_func(name: str):
    return {"sequential": sequential, "simultaneous": simultaneous}[name]
