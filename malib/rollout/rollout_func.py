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
import ray
from collections import defaultdict

import numpy as np

from malib.utils.metrics import get_metric, Metric
from malib.utils.typing import AgentID, Dict, PolicyID, Union
from malib.utils.preprocessor import get_preprocessor
from malib.envs.agent_interface import AgentInterface
from malib.backend.datapool.offline_dataset_server import Episode, MultiAgentEpisode


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
    env: type,
    num_episodes: int,
    agent_interfaces: Dict[AgentID, AgentInterface],
    fragment_length: int,
    behavior_policies: Dict[AgentID, PolicyID],
    agent_episodes: Dict[AgentID, Episode],
    metric: Metric,
    send_interval: int = 50,
    dataset_server: ray.ObjectRef = None,
):
    """Rollout in simultaneous mode, support environment vectorization.

    :param type env: The environment instance.
    :param int num_episodes: Episode number.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param int fragment_length: The maximum step limitation of environment rollout.
    :param Dict[AgentID,PolicyID] behavior_policies: The behavior policy mapping for policy
        specifing when execute `compute_action` `transform_observation. Furthermore, the specified policy id will replace the existing behavior policy if you've reset it before.
    :param Dict[AgentID,Episode] agent_episodes: A dict of agent episodes, for individual experience buffering.
    :param Metric metric: The metric handler, for statistics parsing and grouping.
    :param int send_interval: Specifying the step interval of sending buffering data to remote offline dataset server.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A tuple of statistics and the size of buffered experience.
    """

    rets = env.reset(limits=num_episodes)
    done = False
    cnt = 0

    agent_ids = list(agent_interfaces.keys())

    rets[Episode.CUR_OBS] = dict(
        zip(
            agent_ids,
            ray.get(
                [
                    agent_interfaces[aid].transform_observation.remote(
                        rets[Episode.CUR_OBS][aid], policy_id=behavior_policies[aid]
                    )
                    for aid in agent_ids
                ]
            ),
        )
    )

    while not done and cnt < fragment_length:
        act_dict = {}
        act_dist_dict = {}

        tasks = []
        for aid in agent_ids:
            obs_seq = rets[Episode.CUR_OBS][aid]
            extra_kwargs = {"policy_id": behavior_policies[aid]}
            if rets.get(Episode.ACTION_MASKS) is not None:
                extra_kwargs["action_mask"] = rets[Episode.ACTION_MASKS][aid]
            tasks.append(
                agent_interfaces[aid].compute_action.remote(obs_seq, **extra_kwargs)
            )
        prets = ray.get(tasks)
        for aid, (x, y, _) in zip(agent_ids, prets):
            act_dict[aid] = x
            act_dist_dict[aid] = y

        next_rets = env.step(act_dict)
        rets.update(next_rets)

        for k, v in rets.items():
            if k == Episode.NEXT_OBS:
                tasks = []
                for aid in agent_ids:
                    tasks.append(
                        agent_interfaces[aid].transform_observation.remote(
                            v[aid], policy_id=behavior_policies[aid]
                        )
                    )
                tmpv = ray.get(tasks)
                for aid, e in zip(agent_ids, tmpv):
                    v[aid] = e

        # stack to episodes
        for aid in agent_episodes:
            episode = agent_episodes[aid]
            items = {
                Episode.ACTIONS: np.stack(act_dict[aid]),
                Episode.ACTION_DIST: np.stack(act_dist_dict[aid]),
            }

            for k, v in rets.items():
                items[k] = np.stack(v[aid])

            episode.insert(**items)
            metric.step(aid, behavior_policies[aid], **items)

        rets[Episode.CUR_OBS] = rets[Episode.NEXT_OBS]
        done = any(
            [
                any(v) if not isinstance(v, bool) else v
                for v in rets[Episode.DONES].values()
            ]
        )
        cnt += 1

        if dataset_server is not None and cnt % send_interval == 0:
            dataset_server.save.remote(agent_episodes)
            # clean agent episode
            for e in agent_episodes.values():
                e.reset()

    dataset_server.save.remote(agent_episodes)
    transition_size = cnt * len(agent_episodes) * getattr(env, "num_envs", 1)
    # print("transition size:", transition_size)
    return [metric.parse(agent_filter=list(agent_episodes.keys()))], transition_size


def rollout_wrapper(
    agent_episodes: Union[MultiAgentEpisode, Dict[AgentID, Episode]] = None,
    rollout_type="sequential",
):
    """Rollout wrapper accept a dict of episodes outside.

    :param Union[MultiAgentEpisode,Dict[AgentID,Episode]] agent_episodes: A dict of agent episodes or multiagentepisode instance.
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
        if isinstance(agent_episodes, MultiAgentEpisode):
            agent_episodes.insert(**episodes)
        elif isinstance(agent_episodes, Dict):
            for agent, episode in episodes.items():
                agent_episodes[agent].insert(**episode.data)
        return statistic, episodes

    return func


def get_func(name: str):
    return {"sequential": sequential, "simultaneous": simultaneous}[name]
