"""
Implementation of async rollout worker.
"""
from collections import defaultdict
from malib.algorithm.common.policy import Policy

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
from malib.utils.typing import (
    AgentID,
    Any,
    Dict,
    BehaviorMode,
    PolicyID,
    Tuple,
    Sequence,
    List,
)


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

        parallel_num = kwargs["parallel_num"]
        if remote:
            assert (
                parallel_num > 0
            ), f"parallel_num should be positive, while parallel_num={parallel_num}"

            resources = kwargs.get(
                "resources",
                {
                    "num_cpus": None,
                    "num_gpus": None,
                    "memory": None,
                    "object_store_memory": None,
                    "resources": None,
                },
            )

            Stepping = rollout_func.Stepping.as_remote(**resources)
            self.actors = [
                Stepping.remote(kwargs["exp_cfg"], env_desc, self._offline_dataset)
                for _ in range(parallel_num)
            ]
            self.actor_pool = ActorPool(self.actors)
        else:
            raise NotImplementedError

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.

        :param Dict[AgentID,Dict[PolicyID,float]] policy_distribution: The agent policy distribution
        """

        return BaseRolloutWorker.ready_for_sample(self, policy_distribution)

    def sample(
        self,
        callback: type,
        num_episodes: int,
        fragment_length: int,
        role: str,
        policy_combinations: List,
        explore: bool = True,
        threaded: bool = True,
        policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None,
    ) -> Tuple[Sequence[Dict], Sequence[Any]]:
        """Sample function. Support rollout and simulation. Default in threaded mode."""

        if explore:
            for interface in self._agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLORATION)
        else:
            for interface in self._agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLOITATION)

        if role == "simulation":
            tasks = [
                {"num_episodes": num_episodes, "behavior_policies": comb}
                for comb in policy_combinations
            ]
        elif role == "rollout":
            seg_num = len(self.actors)
            x = num_episodes // seg_num
            y = num_episodes - seg_num * x
            episode_segs = [x] * seg_num + ([y] if y else [])
            assert len(policy_combinations) == 1
            # FIXME(ming): here the policy combinations[0] is actually produced from the trainiable pairs.
            #   so we need to init behavior policies for other fixed agents
            assert policy_distribution is not None
            tasks = [
                {
                    "num_episodes": episode,
                    "behavior_policies": policy_combinations[0],
                    "policy_distribution": policy_distribution,
                }
                for episode in episode_segs
            ]
        else:
            raise TypeError(f"Unkown role: {role}")

        if threaded:
            rets = self.actor_pool.map(
                lambda a, task: a.run.remote(
                    agent_interfaces=self._agent_interfaces,
                    metric_type=self._metric_type,
                    fragment_length=fragment_length,
                    desc=task,
                    callback=callback,
                    role=role,
                ),
                tasks,
            )
        else:
            raise NotImplementedError

        num_frames, stats_list = 0, []
        for ret in rets:
            stats_list.append(ret[0])
            num_frames += ret[1]

        return stats_list, num_frames

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
