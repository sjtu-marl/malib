from typing import Any, Dict, Set

import ray

from malib.common.manager import Manager
from malib.rl.config import Algorithm
from malib.utils.logging import Logger
from malib.rollout.inference.client import InferenceClient


class InferenceManager(Manager):
    def __init__(
        self,
        group_info: Dict[str, Set],
        model_entry_point: Dict[str, str],
        algorithm: Algorithm,
        verbose: bool = False,
        ray_actor_namespace: str = "inference",
    ):
        super().__init__(verbose, namespace=ray_actor_namespace)

        inference_remote_cls = InferenceClient.as_remote(num_cpus=1)
        obs_spaces = group_info["observation_space"]
        act_spaces = group_info["action_space"]
        agent_groups = group_info["agent_groups"]

        self._infer_clients = {}
        self._inference_entry_points = {}
        model_entry_point = (
            model_entry_point
            if model_entry_point is not None
            else {rid: None for rid in agent_groups.keys()}
        )

        infer_client_ready_check = []
        for rid, _ in agent_groups.items():
            actor_name = f"inference_{rid}"
            self._infer_clients[rid] = inference_remote_cls.options(
                namespace=self.namespace, name=actor_name
            ).remote(
                model_entry_point=model_entry_point[rid],
                policy_cls=algorithm.policy,
                observation_space=obs_spaces[rid],
                action_space=act_spaces[rid],
                model_config=algorithm.model_config,
            )
            infer_client_ready_check.append(self._infer_clients[rid].ready.remote())
            self._inference_entry_points[rid] = "{}:{}".format(
                self.namespace, actor_name
            )

        # check ready
        while len(infer_client_ready_check):
            _, infer_client_ready_check = ray.wait(
                infer_client_ready_check, num_returns=1, timeout=1
            )

        Logger.info("All inference clients are ready for serving")

    def get_inference_client(self, runtime_id: str) -> InferenceClient:
        return self.inference_clients[runtime_id]

    @property
    def inference_clients(self) -> Dict[str, ray.ObjectRef]:
        return self._infer_clients

    @property
    def inference_entry_points(self) -> Dict[str, str]:
        """Return a mapping of inference client entrypoints.

        Returns:
            Dict[str, str]: A dict mapping from runtime id to entrypoints.
        """

        return self._inference_entry_points

    def submit(self, task: Any, wait: bool = False):
        pass

    def retrive_results(self):
        pass
