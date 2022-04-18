"""
Implementation of async rollout worker.
"""

import operator

import numpy as np

from functools import reduce
from ray.util import ActorPool

from malib.envs.agent_interface import AgentInterface
from malib.rollout import rollout_func
from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    Any,
    BufferDescription,
    Dict,
    PolicyID,
    Tuple,
    Sequence,
    List,
    Callable,
    TaskDescription,
    Union,
)
from malib.utils.general import iter_many_dicts_recursively


def _parse_rollout_info(raw_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    holder = {}
    for history, ds, k, vs in iter_many_dicts_recursively(*raw_statistics, history=[]):
        prefix = "/".join(history)
        vs = reduce(operator.add, vs)
        holder[f"{prefix}_mean"] = np.mean(vs)
        holder[f"{prefix}_max"] = np.max(vs)
        holder[f"{prefix}_min"] = np.min(vs)
    return holder


class RolloutWorker(BaseRolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def __init__(
        self,
        worker_index: Any,
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable,
        runtime_configs: Dict[str, Any],
        experiment_config: Dict[str, Any],
    ):

        """Create a rollout worker instance.

        :param Any worker_index: Indicates rollout worker
        :param Dict[str,Any] env_desc: The environment description
        :param bool remote: Indicates this rollout worker work in remote mode or not, default by False
        """

        BaseRolloutWorker.__init__(
            self,
            worker_index,
            env_desc,
            agent_mapping_func,
            runtime_configs,
            experiment_config,
        )

    def step_rollout(
        self,
        n_step: int,
        task_desc: TaskDescription,
        buffer_desc: Union[Dict[AgentID, BufferDescription], BufferDescription],
        runtime_configs_template: Dict[str, Any],
    ):
        num_episodes = task_desc.content.num_episodes
        num_rollout_tasks = (
            num_episodes // self.worker_runtime_configs["num_env_per_thread"]
        )

        tasks = [runtime_configs_template for _ in range(num_rollout_tasks)]

        # add tasks for evaluation
        eval_runtime_configs = runtime_configs_template.copy()
        eval_runtime_configs["flag"] = "evaluation"
        tasks.extend(
            [
                eval_runtime_configs
                for _ in range(self.worker_runtime_configs["num_eval_threads"])
            ]
        )

        rets = self.actor_pool.map(
            lambda a, task: a.run.remote(
                agent_interfaces=self.agent_interfaces,
                desc=task,
                buffer_desc=buffer_desc,
            ),
            tasks,
        )

        stats_list = []
        for ret in rets:
            # we retrieve only results from evaluation/simulation actors.
            if ret["task_type"] == "evaluation":
                stats_list.append(ret["eval_info"])
            # if ret["task_type"] == "rollout":
            #     num_frames += ret["total_fragment_length"]

        holder = _parse_rollout_info(stats_list)

        return holder

    def step_simulation(self, task_desc: TaskDescription):
        # set state here
        combinations = task_desc.content.policy_combinations
        num_episodes = task_desc.content.num_episodes
        policy_combinations = [
            {k: p for k, (p, _) in comb.items()} for comb in combinations
        ]

        tasks = [
            {
                "num_episodes": num_episodes,
                "behavior_policies": comb,
                "flag": "simulation",
            }
            for comb in policy_combinations
        ]

        rets = self.actor_pool.map(
            lambda a, task: a.run.remote(
                agent_interfaces=self._agent_interfaces,
                desc=task,
                buffer_desc=None,
            ),
            tasks,
        )

        num_frames, stats_list = 0, []
        for ret in rets:
            # we retrieve only results from evaluation/simulation actors.
            if ret["task_type"] == "simulation":
                stats_list.append(ret["eval_info"])
            # and total fragment length tracking from rollout actors
            if ret["task_type"] == "rollout":
                num_frames += ret["total_fragment_length"]

        return stats_list

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.stop.remote()
            actor.__ray_terminate__.remote()
