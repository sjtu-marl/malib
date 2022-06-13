"""
A `RolloutWorkerManager` contains a cluster of `RolloutWorker` (in the future version, each worker will be wrapped in a
subprocess). It is responsible for the resources management of worker instances, also statistics collections. Workers
will be assigned with rollout tasks sent from the `CoordinatorServer`.
"""

from argparse import Namespace
import traceback
from typing import Dict, Tuple, Any, Callable, Set, List
from collections import defaultdict

import hashlib
import time

import ray
import numpy as np

from ray.util import ActorPool

from malib.common.manager import Manager
from malib.common.strategy_spec import StrategySpec
from malib.rollout.rollout_worker import RolloutWorker
from malib.rollout.ray_inference.client import RayInferenceClient
from malib.rollout.ray_inference.server import RayInferenceWorkerSet
from malib.utils.logging import Logger


def _get_worker_hash_idx(idx):
    hash_coding = hashlib.md5()
    hash_coding.update(bytes(f"worker-{idx}-{time.time()}", "utf-8"))
    return hash_coding.hexdigest()


def validate_strategy_specs(specs: Dict[str, StrategySpec]):
    for rid, spec in specs.items():
        if len(spec) < 1:
            raise ValueError(f"Empty spec for runtime_id={rid}")
        # check prob list
        expected_prob_list = spec.meta_data.get(
            "prob_list", [1 / len(spec)] * len(spec)
        )
        if expected_prob_list is None:
            raise ValueError(
                f"donot give an empty prob list explictly for runtime_id={rid}."
            )
        if not np.isclose(sum(expected_prob_list), 1.0):
            raise ValueError(
                f"The summation of prob list for runtime_id={rid} shoud be close to 1.: {expected_prob_list}."
            )


class RolloutWorkerManager(Manager):
    def __init__(
        self,
        experiment_tag: str,
        stopping_conditions: Dict[str, Any],
        num_worker: int,
        agent_mapping_func: Callable,
        rollout_config: Dict[str, Any],
        env_desc: Dict[str, Any],
        log_dir: str,
        resource_config: Dict[str, Any] = None,
    ):
        """Construct a manager for multiple rollout workers.

        Args:
            experiment_tag (str): Experiment tag.
            num_worker (int): Indicates how many rollout workers will be initialized.
            agent_mapping_func (Callable): Agent mapping function, maps agents to runtime id.
            rollout_config (Dict[str, Any]): Runtime rollout configuration.
            env_desc (Dict[str, Any]): Environment description.
            log_dir (str): Log directory.
            resource_config (Dict[str, Any], optional): A dict that describes the resource config. Defaults to None.
        """

        super().__init__()

        rollout_worker_cls = RolloutWorker
        worker_cls = rollout_worker_cls.as_remote(
            num_cpus=None,
            num_gpus=None,
            memory=None,
            object_store_memory=None,
            resources=None,
        )
        workers = []

        for i in range(num_worker):
            workers.append(
                worker_cls.options(max_concurrency=100).remote(
                    experiment_tag=experiment_tag,
                    env_desc=env_desc,
                    agent_mapping_func=agent_mapping_func,
                    runtime_config=rollout_config,
                    log_dir=log_dir,
                    reverb_table_kwargs={},
                    rollout_callback=None,
                    simulate_callback=None,
                )
            )

        self._workers: List[ray.actor] = workers
        self._actor_pool = ActorPool(self._workers)

        agent_groups = defaultdict(lambda: set())
        for agent in env_desc["possible_agents"]:
            rid = agent_mapping_func(agent)
            agent_groups[rid].add(agent)
        self._runtime_ids = tuple(agent_groups.keys())
        self._agent_groups = dict(agent_groups)

        # for debug
        self.observation_spaces = env_desc["observation_spaces"]
        self.action_spaces = env_desc["action_spaces"]
        self.experiment_tag = experiment_tag

        assert (
            "rollout" in stopping_conditions
        ), f"Stopping conditions should contain `rollout`: {stopping_conditions}"
        self.stopping_conditions = stopping_conditions

    @property
    def runtime_ids(self) -> Tuple[str]:
        return self._runtime_ids

    @property
    def agent_groups(self) -> Dict[str, Set]:
        return self._agent_groups

    def simulate(self, task_list):
        """Parse simulation task and dispatch it to available workers"""

        self._actor_pool.map(
            lambda actor, task: actor.simulate.remote(
                runtime_strategy_specs=task.strategy_specs,
                trainable_agents=task.trainable_agents,
            ),
            task_list,
        )

    def rollout(self, task_list) -> None:
        """Start rollout task without blocking.

        Args:
            task_list (_type_): A list of task dict, keys include:
                - `strategy_specs`: a dict of strategy specs, mapping from runtime ids to specs.
                - `trainable_agents`: a list of trainable agents.

        """

        # validate all strategy specs here
        for task in task_list:
            validate_strategy_specs(task["strategy_specs"])

        Logger.debug(f"accept rollout tasks as: {task_list}")

        self.pending_tasks = self._actor_pool.map(
            lambda actor, task: actor.rollout.remote(
                runtime_strategy_specs=task["strategy_specs"],
                stopping_conditions=self.stopping_conditions["rollout"],
                trainable_agents=task["trainable_agents"],
                data_entrypoints=task["data_entrypoints"],
            ),
            task_list,
        )

    def wait(self):
        try:
            for task in self.pending_tasks:
                print("done or not:", task)
                if self._force_stop:
                    self.terminate()
                    break
            print("seems done", self._force_stop)
        except Exception:
            traceback.print_exc()

    def terminate(self):
        """Stop all remote workers"""

        for worker in self._workers.values():
            worker.close.remote()
            ray.kill(worker)
