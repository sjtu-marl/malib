# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import operator

import numpy as np

from functools import reduce

from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    Any,
    BufferDescription,
    Dict,
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
        tasks = [
            runtime_configs_template
            for _ in range(self.worker_runtime_configs["num_threads"])
        ]

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
        num_frames = 0
        for ret in rets:
            # we retrieve only results from evaluation/simulation actors.
            if ret["task_type"] == "evaluation":
                stats_list.append(ret["eval_info"])
            num_frames += ret["total_fragment_length"]

        holder = _parse_rollout_info(stats_list)

        return holder, num_frames

    def step_simulation(self, task_desc: TaskDescription):
        # TODO(ming): has not been tested yet.
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
                agent_interfaces=self.agent_interfaces,
                desc=task,
                buffer_desc=None,
            ),
            tasks,
        )

        stats_list = []
        for ret in rets:
            stats_list.append(ret["eval_info"])
        return stats_list

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.stop.remote()
            actor.__ray_terminate__.remote()
