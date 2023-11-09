from typing import Dict, Set

import ray

from malib.common.manager import Manager
from malib.scenarios import Scenario
from malib.rollout.inference.client import InferenceClient


class InferenceManager(Manager):
    def __init__(
        self,
        group_info: Dict[str, Set],
        ray_actor_namespace: str,
        entrypoints: Dict[str, str],
        scenario: Scenario,
        verbose: bool = False,
    ):
        super().__init__(verbose, namespace=ray_actor_namespace)

        inference_remote_cls = InferenceClient.as_remote(num_cpus=1).options(
            namespace=self.namespace
        )
        obs_spaces = group_info["observation_space"]
        act_spaces = group_info["action_space"]
        agent_groups = group_info["agent_groups"]

        self.infer_clients = {}
        for rid, _ in agent_groups.items():
            self.infer_clients[rid] = inference_remote_cls.options(
                name=f"inference_{rid}"
            ).remote(
                entry_point=entrypoints[rid],
                policy_cls=scenario.algorithms[rid].policy_cls,
                observation_space=obs_spaces[rid],
                action_space=act_spaces[rid],
                model_config=scenario.training_config["model_config"],
            )

        # check ready
        tasks = list(self.infer_clients.values())
        while len(tasks):
            _, tasks = ray.wait(tasks, num_returns=1, timeout=1)
