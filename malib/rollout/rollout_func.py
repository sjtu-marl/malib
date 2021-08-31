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

from typing import Callable
import uuid

import ray
import numpy as np

from collections import defaultdict

from malib import settings
from malib.utils.logger import get_logger, Log
from malib.utils.metrics import get_metric, Metric
from malib.utils.typing import (
    AgentID,
    Dict,
    MetricEntry,
    PolicyID,
    RolloutConfig,
    Union,
    Any,
    Tuple,
)
from malib.utils.preprocessor import get_preprocessor
from malib.envs import Environment
from malib.envs.agent_interface import AgentInterface
from malib.envs.vector_env import VectorEnv
from malib.backend.datapool.offline_dataset_server import (
    Episode,
    SequentialEpisode,
)


def sequential(
    env: Environment,
    num_episodes: int,
    agent_interfaces: Dict[AgentID, AgentInterface],
    fragment_length: int,
    behavior_policies: Dict[AgentID, PolicyID],
    agent_episodes: Dict[AgentID, Episode],
    metric: Metric,
    send_interval: int = 50,
    dataset_server: ray.ObjectRef = None,
):
    """ Rollout in sequential manner """

    # use env.env as real env
    env = env.env
    cnt = 0
    evaluated_results = []

    assert fragment_length > 0, fragment_length
    for ith in range(num_episodes):
        env.reset()
        metric.reset()
        for aid in env.agent_iter(max_iter=fragment_length):
            observation, reward, done, info = env.last()

            if isinstance(observation, dict):
                info = {"action_mask": np.asarray([observation["action_mask"]])}
                action_mask = observation["action_mask"]
            else:
                action_mask = np.ones(
                    get_preprocessor(env.action_spaces[aid])(
                        env.action_spaces[aid]
                    ).size
                )

            # observation has been transferred
            observation = agent_interfaces[aid].transform_observation(
                observation, behavior_policies[aid]
            )

            info["policy_id"] = behavior_policies[aid]

            if not done:
                action, action_dist, extra_info = agent_interfaces[aid].compute_action(
                    [observation], **info
                )
                # convert action to scalar
                action = action[0]
            else:
                info["policy_id"] = behavior_policies[aid]
                action = None
            env.step(action)
            if action is None:
                action = [agent_interfaces[aid].action_space.sample()]
            if aid in agent_episodes:
                agent_episodes[aid].insert(
                    **{
                        Episode.CUR_OBS: [observation],
                        Episode.ACTION_MASK: [action_mask],
                        Episode.ACTION_DIST: action_dist,
                        Episode.ACTION: action,
                        Episode.REWARD: reward,
                        Episode.DONE: done,
                    }
                )
            metric.step(
                aid,
                behavior_policies[aid],
                observation=observation,
                action=action,
                action_dist=action_dist,
                reward=reward,
                done=done,
                info=info,
            )
            cnt += 1
        evaluated_results.append(
            metric.parse(agent_filter=tuple(agent_episodes.keys()))
        )
        # when dataset_server is not None
    if dataset_server:
        for e in agent_episodes.values():
            e.clean_data()
        dataset_server.save.remote(agent_episodes, wait_for_ready=False)
        for e in agent_episodes.values():
            e.reset()

    # aggregated evaluated results groupped in agent wise
    evaluated_results = metric.merge_parsed(evaluated_results)
    return evaluated_results, cnt


def simultaneous(
    env: type,
    num_envs: int,
    agent_interfaces: Dict[AgentID, AgentInterface],
    fragment_length: int,
    max_step: int,
    behavior_policies: Dict[AgentID, PolicyID],
    agent_episodes: Dict[AgentID, Episode],
    metric: Metric,
    send_interval: int = 50,
    dataset_server: ray.ObjectRef = None,
):
    """Rollout in simultaneous mode, support environment vectorization.

    :param type env: The environment instance.
    :param int num_envs: The number of parallel environments.
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

    rets = env.reset(
        limits=num_envs,
        fragment_length=fragment_length,
        env_reset_kwargs={"max_step": max_step},
    )

    # metric.reset(mode="vector")
    episode_length = []
    episode_rewards = {aid: [] for aid in agent_episodes}

    cnt = 0

    observations = {
        aid: agent_interfaces[aid].transform_observation(obs)
        for aid, obs in rets[Episode.CUR_OBS].items()
    }

    metric.reset(mode="vector")

    while not env.is_terminated():
        actions, action_dists, action_masks = {}, {}, {}
        for aid, observation in observations.items():
            action_masks[aid] = (
                rets[Episode.ACTION_MASK][aid]
                if rets.get(Episode.ACTION_MASK) is not None
                else None
            )
            actions[aid], action_dists[aid], _ = agent_interfaces[aid].compute_action(
                observation,
                policy_id=behavior_policies[aid],
                action_mask=action_masks[aid],
            )
        rets = env.step(actions)

        if "total_rewards" in rets:
            for e in rets["total_rewards"]:
                for aid, v in e.items():
                    episode_rewards[aid].append(v)
        if "cnt" in rets:
            episode_length.extend(rets["cnt"])

        next_observations = {
            aid: agent_interfaces[aid].transform_observation(obs)
            for aid, obs in rets[Episode.CUR_OBS].items()
        }

        if dataset_server:
            for aid in env.trainable_agents:
                agent_episodes[aid].insert(
                    **{
                        Episode.CUR_OBS: observations[aid],
                        Episode.NEXT_OBS: next_observations[aid],
                        Episode.REWARD: rets[Episode.REWARD][aid],
                        Episode.ACTION: actions[aid],
                        Episode.DONE: rets[Episode.DONE][aid],
                        Episode.ACTION_DIST: action_dists[aid],
                    }
                )
        observations = next_observations
        cnt += 1

    if dataset_server:
        dataset_server.save.remote(agent_episodes, wait_for_ready=False)
    transition_size = cnt * len(agent_episodes) * getattr(env, "num_envs", 1)

    evaluated_results = metric.parse(agent_filter=tuple(agent_episodes.keys()))
    return evaluated_results, transition_size


def get_func(name: Union[str, Callable]):
    if callable(name):
        return name
    else:
        return {"sequential": sequential, "simultaneous": simultaneous}[name]


class Stepping:
    def __init__(
        self, exp_cfg: Dict[str, Any], env_desc: Dict[str, Any], dataset_server=None
    ):
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name=f"rolloutfunc_executor_{uuid.uuid1()}",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )

        # init environment here
        self.env_desc = env_desc

        # check whether env is simultaneous
        env = env_desc["creator"](**env_desc["config"])
        self._is_sequential = env.is_sequential

        if not env.is_sequential:
            self.env = VectorEnv.from_envs([env], config=env_desc["config"])
            self.callback = simultaneous
        else:
            self.env = env
            self.callback = sequential

        self._dataset_server = dataset_server

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization."""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces: Dict[AgentID, AgentInterface],
        metric_type: Union[str, type],
        fragment_length: int,
        desc: Dict[str, Any],
        callback: type,
        role: str,
        episode_buffers: Dict[AgentID, Episode] = None,
    ) -> Tuple[Dict[str, MetricEntry], int]:
        """Environment stepping, rollout/simulate with environment vectorization if it is feasible.

        :param Dict[AgentID,AgentInterface] agent_interface: A dict of agent interfaces.
        :param Union[str,type] metric_type: Metric type or handler.
        :param int fragment_length: The maximum length of an episode.
        :param Dict[str,Any] desc: The description of task.
        :param type callback: Customized/registered rollout function.
        :param str role: Indicator of stepping type. Values in `rollout` or `simulation`.
        :returns: A tuple of a dict of MetricEntry and the caculation of total frames.
        """

        behavior_policies = {}
        # desc: policy_distribution, behavior_policies, num_episodes
        policy_distribution = desc.get("policy_distribution")
        for agent, interface in agent_interfaces.items():
            if policy_distribution:
                interface.reset(policy_distribution[agent])
            behavior_policies[agent] = interface.behavior_policy

        # behavior policies is a mapping from agents to policy ids
        # update with external behavior_policies
        behavior_policies.update(desc["behavior_policies"])
        # specify the number of running episodes
        num_episodes = desc["num_episodes"]
        max_step = desc.get("max_step", 1000)

        self.add_envs(num_episodes)

        if role == "rollout":
            episode_creator = Episode if not self._is_sequential else SequentialEpisode
            episode_buffers = episode_buffers or {
                agent: episode_creator(
                    self.env_desc["config"]["env_id"],
                    desc["behavior_policies"][agent],
                    capacity=self._rollout_config["fragment_length"],
                    other_columns=self.env.extra_returns,
                )
                for agent in desc["behavior_policies"]
            }
        else:
            episode_buffers = None

        metric = get_metric(metric_type)(self.env.possible_agents)
        callback = get_func(callback) if callback else self.callback

        evaluated_results, num_frames = callback(
            self.env,
            num_episodes,
            agent_interfaces,
            fragment_length,
            max_step,
            behavior_policies,
            episode_buffers,
            metric,
            dataset_server=self._dataset_server if role == "rollout" else None,
        )

        return evaluated_results, num_frames

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these new environment instances into it,
        otherwise do nothing.

        :returns: The number of nested environments.
        """

        if not isinstance(self.env, VectorEnv):
            return 1

        existing_env_num = getattr(self.env, "num_envs", 1)

        if existing_env_num >= maximum:
            return self.env.num_envs

        self.env.add_envs(num=maximum - existing_env_num)

        return self.env.num_envs

    def close(self):
        if self.env is not None:
            self.env.close()
