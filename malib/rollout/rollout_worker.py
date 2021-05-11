"""
Implementation of async rollout worker.
"""
import gc
import time
from collections import defaultdict

import ray
from ray.util import ActorPool

import uuid
from malib import settings
from malib.backend.datapool.offline_dataset_server import Episode, MultiAgentEpisode
from malib.envs.agent_interface import AgentInterface
from malib.rollout import rollout_func
from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.logger import Log, get_logger
from malib.utils.typing import Any, Dict, BehaviorMode, Tuple, Sequence


class Func:
    def __init__(self, exp_cfg):
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name=f"rolloutfunc_executor_{uuid.uuid1()}",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )

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
        trainable_pairs,
        agent_interfaces,
        env_desc,
        metric_type,
        max_iter,
        policy_mapping,
        num_episode,
        callback,
        role="rollout",
    ):
        ith = 0
        statics, data = [], []
        if isinstance(callback, str):
            callback = rollout_func.get_func(callback)
        env = env_desc["creator"](**env_desc["config"])
        env_desc["env"] = env
        while ith < num_episode:
            tmp_statistic, tmp_data = callback(
                trainable_pairs=trainable_pairs,
                agent_interfaces=agent_interfaces,
                env_desc=env_desc,
                metric_type=metric_type,
                max_iter=max_iter,
                behavior_policy_mapping=policy_mapping,
            )
            statics.append(tmp_statistic)
            if role == "rollout":
                data.append(tmp_data)
            ith += 1

        merged_data = defaultdict(list)
        merged_capacity = defaultdict(lambda: 0)
        for d in data:
            for aid, episode in d.items():
                merged_data[aid].append(episode)
                merged_capacity[aid] += episode.size

        data2send = {
            aid: Episode.concatenate(*merged_data[aid], capacity=merged_capacity[aid])
            for aid in merged_data
        }
        # env.close()
        del env
        return statics, data2send


class RolloutWorker(BaseRolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def __init__(
        self,
        worker_index: Any,
        env_desc: Dict[str, Any],
        metric_type: str,
        remote: bool = False,
        **kwargs,
    ):

        """Create a rollout worker instance.

        :param Any worker_index: Indicates rollout worker
        :param Dict[str,Any] env_desc: The environment description
        :param str metric_type: Name of registered metric handler
        :param bool remote: Indicates this rollout worker work in remote mode or not, default by False
        """

        BaseRolloutWorker.__init__(
            self, worker_index, env_desc, metric_type, remote, **kwargs
        )

        parallel_num = kwargs.get("parallel_num", 0)
        if parallel_num:
            RemoteFunc = Func.as_remote()
            self.actors = [
                RemoteFunc.remote(kwargs["exp_cfg"]) for _ in range(parallel_num)
            ]
            self.actor_pool = ActorPool(self.actors)

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.

        :param Dict[AgentID,Dict[PolicyID,float]] policy_distribution: The agent policy distribution
        """

        return BaseRolloutWorker.ready_for_sample(self, policy_distribution)

    def _rollout(
        self, threaded, episode_seg, **kwargs
    ) -> Tuple[Sequence[Dict], Sequence[Any]]:
        """Helper function to support rollout."""

        if threaded:
            stat_data_tuples = self.actor_pool.map_unordered(
                lambda a, v: a.run.remote(
                    num_episode=v,
                    **kwargs,
                ),
                episode_seg,
            )
        else:
            stat_data_tuples = []
            for v in episode_seg:
                statistics, data = Func.run(None, num_episode=v, **kwargs)
                stat_data_tuples.append((statistics, data))
        statistic_seq = []
        merged_data = defaultdict(list)
        merged_capacity = defaultdict(lambda: 0)
        for statis, data in stat_data_tuples:
            for aid, episode in data.items():
                merged_data[aid].append(episode)
                merged_capacity[aid] += episode.size
            statistic_seq.append(statis)

        agent_episode = {
            aid: Episode.concatenate(*merged_data[aid], capacity=merged_capacity[aid])
            for aid in merged_data
        }
        data2send = {
            aid: MultiAgentEpisode(
                e.env_id,
                kwargs["trainable_pairs"],
                merged_capacity[aid],
                e.other_columns,
            )
            for aid, e in agent_episode.items()
        }
        for aid, mae in data2send.items():
            mae.insert(**agent_episode)
        return statistic_seq, data2send

    def _simulation(self, threaded, combinations, **kwargs):
        """Helper function to support simulation."""

        if threaded:
            print(f"got simulation task: {len(combinations)}")
            res = self.actor_pool.map(
                lambda a, combination: a.run.remote(
                    trainable_pairs=None,
                    policy_mapping={aid: v[0] for aid, v in combination.items()},
                    **kwargs,
                ),
                combinations,
            )
            # depart res into two parts
            statis = []
            statis = [e[0] for e in res]
            return statis, None
        else:
            statis = []
            for comb in combinations:
                tmp, _ = Func.run(
                    None,
                    trainable_pairs=None,
                    policy_mapping={aid: v[0] for aid, v in comb.items()},
                    **kwargs,
                )
                statis.append(tmp)
            return statis, None

    def sample(self, *args, **kwargs) -> Tuple[Sequence[Dict], Sequence[Any]]:
        """Sample function. Support rollout and simulation. Default in threaded mode."""

        callback = kwargs["callback"]
        behavior_policy_mapping = kwargs.get("behavior_policy_mapping", None)
        num_episodes = kwargs["num_episodes"]
        trainable_pairs = kwargs.get("trainable_pairs", None)
        threaded = kwargs.get("threaded", True)
        explore = kwargs.get("explore", True)
        fragment_length = kwargs.get("fragment_length", 1000)
        role = kwargs["role"]  # rollout or simulation

        if explore:
            for interface in self._agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLORATION)
        else:
            for interface in self._agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLOITATION)

        if role == "rollout":
            return self._rollout(
                threaded,
                num_episodes,
                trainable_pairs=trainable_pairs,
                agent_interfaces=self._agent_interfaces,
                env_desc=self._env_description,
                metric_type=self._metric_type,
                max_iter=fragment_length,
                policy_mapping=behavior_policy_mapping,
                callback=callback,
                role="rollout",
            )
        else:
            return self._simulation(
                threaded,
                behavior_policy_mapping,
                agent_interfaces=self._agent_interfaces,
                env_desc=self._env_description,
                metric_type=self._metric_type,
                max_iter=fragment_length,
                callback=callback,
                num_episode=num_episodes,
                role="simulation",
            )

    # @Log.method_timer(enable=False)
    def update_population(self, agent, policy_id, policy):
        """ Update population with an existing policy instance """

        agent_interface: AgentInterface = self._agent_interfaces[agent]
        agent_interface.policies[policy_id] = policy

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.stop.remote()
            actor.__ray_terminate__.remote()
