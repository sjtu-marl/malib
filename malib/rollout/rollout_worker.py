"""
Implementation of async rollout worker.
"""

from ray.util import ActorPool

from malib.envs.agent_interface import AgentInterface
from malib.rollout import rollout_func
from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.typing import (
    AgentID,
    Any,
    BufferDescription,
    Dict,
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
        save: bool = False,
        **kwargs,
    ):

        """Create a rollout worker instance.

        :param Any worker_index: Indicates rollout worker
        :param Dict[str,Any] env_desc: The environment description
        :param str metric_type: Name of registered metric handler
        :param bool remote: Indicates this rollout worker work in remote mode or not, default by False
        """

        BaseRolloutWorker.__init__(
            self, worker_index, env_desc, metric_type, remote, save, **kwargs
        )

        self._parallel_num = kwargs.get("parallel_num", 1)
        self._evaluation_worker_num = kwargs.get("evaluation_worker_num", 1)
        self._resources = kwargs.get(
            "resources",
            {
                "num_cpus": None,
                "num_gpus": None,
                "memory": None,
                "object_store_memory": None,
                "resources": None,
            },
        )

        self.actor_pool = None
        if remote:
            assert (
                self._parallel_num > 0
            ), f"parallel_num should be positive, while parallel_num={self._parallel_num}"

            Stepping = rollout_func.Stepping.as_remote(**self._resources)
            self.actors = [
                Stepping.remote(
                    kwargs["exp_cfg"],
                    env_desc,
                    self._offline_dataset,
                    use_subproc_env=kwargs["use_subproc_env"],
                    batch_mode=kwargs["batch_mode"],
                    postprocessor_type=kwargs["postprocessor_type"],
                )
                for _ in range(self._parallel_num)
            ]
            self.actors.extend(
                [
                    Stepping.remote(
                        kwargs["exp_cfg"],
                        env_desc,
                        None,
                        use_subproc_env=kwargs["use_subproc_env"],
                        batch_mode=kwargs["batch_mode"],
                        postprocessor_type=kwargs["postprocessor_type"],
                    )
                    for _ in range(self._evaluation_worker_num)
                ]
            )
            self.actor_pool = ActorPool(self.actors)

    def check_actor_pool_available(self):
        if self.actor_pool is None:
            # create actor pool
            self.logger.warning(
                "Actor pool has not been created yet, will generate a new one."
            )
            assert (
                self._parallel_num > 0
            ), f"parallel_num should be positive, while parallel_num={self._parallel_num}"

            Stepping = rollout_func.Stepping.as_remote(**self._resources)
            self.actors = [
                Stepping.remote(
                    self._kwargs["exp_cfg"],
                    self._env_description,
                    self._offline_dataset,
                    use_subproc_env=self._kwargs["use_subproc_env"],
                    batch_mode=self._kwargs["batch_mode"],
                    postprocessor_type=self._kwargs["postprocessor_type"],
                )
                for _ in range(self._parallel_num)
            ]
            self.actors.extend(
                [
                    Stepping.remote(
                        self._kwargs["exp_cfg"],
                        self._env_description,
                        None,
                        use_subproc_env=self._kwargs["use_subproc_env"],
                        batch_mode=self._kwargs["batch_mode"],
                        postprocessor_type=self._kwargs["postprocessor_type"],
                    )
                    for _ in self._evaluation_worker_num
                ]
            )
            self.actor_pool = ActorPool(self.actors)

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
        buffer_desc: BufferDescription = None,
    ) -> Tuple[Sequence[Dict[str, List]], int]:
        """Sample function. Support rollout and simulation. Default in threaded mode."""

        if role == "simulation":
            tasks = [
                {
                    "num_episodes": num_episodes,
                    "behavior_policies": comb,
                    "flag": "simulation",
                }
                for comb in policy_combinations
            ]
        elif role == "rollout":
            seg_num = self._parallel_num
            x = num_episodes // seg_num
            y = num_episodes - seg_num * x
            episode_segs = [x] * seg_num + ([y] if y else [])
            assert len(policy_combinations) == 1
            assert policy_distribution is not None
            tasks = [
                {
                    "flag": "rollout",
                    "num_episodes": episode,
                    "behavior_policies": policy_combinations[0],
                    "policy_distribution": policy_distribution,
                }
                for episode in episode_segs
            ]
            # add tasks for evaluation
            tasks.extend(
                [
                    {
                        "flag": "evaluation",
                        "num_episodes": 10,  # FIXME(ziyu): fix it and debug
                        "behavior_policies": policy_combinations[0],
                        "policy_distribution": policy_distribution,
                    }
                    for _ in range(self._evaluation_worker_num)
                ]
            )
        else:
            raise TypeError(f"Unkown role: {role}")

        self.check_actor_pool_available()
        rets = self.actor_pool.map(
            lambda a, task: a.run.remote(
                agent_interfaces=self._agent_interfaces,
                fragment_length=fragment_length,
                desc=task,
                callback=callback,
                buffer_desc=buffer_desc,
            ),
            tasks,
        )

        num_frames, stats_list = 0, []
        for ret in rets:
            # we save only evaluation ret[0] from evaluation workers
            if ret[0] in ["evaluation", "simulation"]:
                stats_list.append(ret[1]["eval_info"])
            if ret[0] == "rollout":
                num_frames += ret[1]["total_fragment_length"]

        return stats_list, num_frames

    # @Log.method_timer(enable=False)
    def update_population(self, agent, policy_id, policy):
        """Update population with an existing policy instance"""

        agent_interface: AgentInterface = self._agent_interfaces[agent]
        agent_interface.policies[policy_id] = policy

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.stop.remote()
            actor.__ray_terminate__.remote()
