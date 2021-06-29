"""
Implementation of async rollout worker.
"""
from collections import defaultdict

import ray
from ray.util import ActorPool

import uuid
from malib import settings
from malib import rollout
from malib.backend.datapool.offline_dataset_server import Episode, MultiAgentEpisode
from malib.envs.agent_interface import AgentInterface
from malib.rollout import rollout_func
from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.logger import Log, get_logger
from malib.utils.typing import Any, Dict, BehaviorMode, Tuple, Sequence


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
            Stepping = rollout_func.Stepping.as_remote()
            self.actors = [
                Stepping.remote(kwargs["exp_cfg"]) for _ in range(parallel_num)
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
                statistics, data = rollout_func.Stepping.run(
                    None, num_episode=v, **kwargs
                )
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
        ap_mapping = {k: v.policy_id for k, v in agent_episode.items()}
        data2send = {
            aid: MultiAgentEpisode(
                e.env_id,
                ap_mapping,
                merged_capacity[aid],
                e.other_columns,
            )
            for aid, e in agent_episode.items()
        }
        for aid, mae in data2send.items():
            mae.insert(**agent_episode)
        return statistic_seq, data2send

    def _simulation(self, threaded, combinations, **kwargs):
        if threaded:
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
                tmp, _ = rollout_func.Stepping.run(
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
