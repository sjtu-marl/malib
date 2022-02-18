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
        save: bool = False,
        **kwargs,
    ):

        """Create a rollout worker instance.

        :param Any worker_index: Indicates rollout worker
        :param Dict[str,Any] env_desc: The environment description
        :param bool remote: Indicates this rollout worker work in remote mode or not, default by False
        """

        BaseRolloutWorker.__init__(self, worker_index, env_desc, save, **kwargs)

        self._num_rollout_actors = kwargs.get("num_rollout_actors", 1)
        self._num_eval_actors = kwargs.get("num_eval_actors", 1)
        # XXX(ming): computing resources for rollout / evaluation
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

        assert (
            self._num_rollout_actors > 0
        ), f"num_rollout_actors should be positive, but got `{self._num_rollout_actors}`"

        Stepping = rollout_func.Stepping.as_remote(**self._resources)
        self.actors = [
            Stepping.remote(
                kwargs["exp_cfg"],
                env_desc,
                self._offline_dataset,
                use_subproc_env=kwargs["use_subproc_env"],
                batch_mode=kwargs["batch_mode"],
                postprocessor_types=kwargs["postprocessor_types"],
            )
            for _ in range(self._num_rollout_actors)
        ]
        self.rollout_actor_pool = ActorPool(self.actors[: self._num_rollout_actors])
        self.actors.extend(
            [
                Stepping.remote(
                    kwargs["exp_cfg"],
                    env_desc,
                    None,
                    use_subproc_env=kwargs["use_subproc_env"],
                    batch_mode=kwargs["batch_mode"],
                    postprocessor_types=kwargs["postprocessor_types"],
                )
                for _ in range(self._num_eval_actors)
            ]
        )
        self.eval_actor_pool = ActorPool(self.actors[self._num_eval_actors :])

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.

        :param Dict[AgentID,Dict[PolicyID,float]] policy_distribution: The agent policy distribution
        """

        return BaseRolloutWorker.ready_for_sample(self, policy_distribution)

    def sample(
        self,
        num_episodes: int,
        fragment_length: int,
        role: str,
        policy_combinations: List,
        policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None,
        buffer_desc: BufferDescription = None,
    ) -> Tuple[Sequence[Dict[str, List]], int]:
        """Sample function, handling rollout or simulation tasks."""

        if role == "simulation":
            tasks = [
                {
                    "num_episodes": num_episodes,
                    "behavior_policies": comb,
                    "flag": "simulation",
                }
                for comb in policy_combinations
            ]
            actor_pool = self.eval_actor_pool
        elif role == "rollout":
            seg_num = self._num_rollout_actors
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
                    for _ in range(self._num_eval_actors)
                ]
            )
            actor_pool = self.rollout_actor_pool
        else:
            raise TypeError(f"Unkown role: {role}")

        # self.check_actor_pool_available()
        rets = actor_pool.map(
            lambda a, task: a.run.remote(
                agent_interfaces=self._agent_interfaces,
                fragment_length=fragment_length,
                desc=task,
                buffer_desc=buffer_desc,
            ),
            tasks,
        )

        num_frames, stats_list = 0, []
        for ret in rets:
            # we retrieve only results from evaluation/simulation actors.
            if ret[0] in ["evaluation", "simulation"]:
                stats_list.append(ret[1]["eval_info"])
            # and total fragment length tracking from rollout actors
            if ret[0] == "rollout":
                num_frames += ret[1]["total_fragment_length"]

        return stats_list, num_frames

    def update_population(self, agent, policy_id, policy):
        """Update population with an existing policy instance"""

        agent_interface: AgentInterface = self._agent_interfaces[agent]
        agent_interface.policies[policy_id] = policy

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.stop.remote()
            actor.__ray_terminate__.remote()
