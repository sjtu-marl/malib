import ray
import time
import gym
import os
import numpy as np

from ray.util.queue import Queue

from malib import settings
from malib.utils.logger import Logger, Log
from malib.utils.typing import (
    AgentID,
    Any,
    BufferDescription,
    DataFrame,
    List,
    PolicyID,
    Dict,
    Status,
    BehaviorMode,
    Tuple,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.episode import Episode, NewEpisodeDict
from malib.algorithm.common.policy import Policy
from malib.envs.vector_env import SubprocVecEnv, VectorEnv
from malib.rollout.postprocessor import get_postprocessor
from malib.rollout.inference_server import InferenceWorkerSet


def process_env_rets(env_rets: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in env_rets.items():
        pass


def process_policy_outputs():
    pass


def postprocessing(episodes, postprocessor_types, policies=None):
    for handler in get_postprocessor(postprocessor_types):
        episodes = handler(episodes, policies)
    return episodes


@ray.remote
class InferenceClient:
    def __init__(
        self,
        env_desc,
        dataset_server,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
    ):
        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()

        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]
        env_cls = env_desc["creator"]
        env_config = env_desc["config"]

        if use_subproc_env:
            self.env = SubprocVecEnv(
                obs_spaces, act_spaces, env_cls, env_config, max_num_envs=2
            )
        else:
            self.env = VectorEnv(obs_spaces, act_spaces, env_cls, env_config)

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces: Dict[AgentID, InferenceWorkerSet],
        fragment_length: int,
        desc: Dict[str, Any],
        buffer_desc: BufferDescription = None,
    ) -> Tuple[str, Dict[str, List]]:

        task_type = desc["flag"]
        # compute behavior_policies here
        server_runtime_config = {
            "behavior_mode": None,
            "main_behavior_policies": desc["behavior_policies"],
            "policy_distribution": desc.get("policy_distribution", None),
            "sample_mode": "once",
            "parameter_desc_dict": desc["paramter_desc_dict"],
        }

        client_runtime_config = {
            "max_step": desc.get("max_step", None),
            "fragment_length": fragment_length,
            "num_envs": desc["num_episodes"],
            "custom_reset_config": None,
            "trainable_mapping": desc["behavior_policies"]
            if task_type == "rollout"
            else None,
        }

        if task_type == "rollout":
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLORATION
        elif task_type in ["evaluation", "simulation"]:
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLOITATION

        # build connection with agent interfaces
        recv_queue, send_queue = Queue(), Queue()
        _ = ray.get(
            [
                server.connect.remote(
                    [recv_queue, send_queue],
                    runtime_config=server_runtime_config,
                    runtime_id=self.process_id,
                )
                for server in agent_interfaces.values()
            ],
            timeout=3.0,
        )

        rets = self.env.reset(
            limits=client_runtime_config["num_envs"],
            fragment_length=client_runtime_config["fragment_length"],
            max_step=client_runtime_config["max_step"],
            custom_reset_config=client_runtime_config["custom_reset_config"],
            trainable_mapping=client_runtime_config["trainable_mapping"],
        )

        # TODO(ming): process env returns here
        rets = process_env_rets(rets)
        episodes = NewEpisodeDict(lambda env_id: Episode(None, env_id=env_id))

        while not self.env.is_terminated():
            # send query to servers
            for agent, server in agent_interfaces.items():
                send_queue.put_nowait(rets[agent])
            while recv_queue.empty:
                time.sleep(1)
            policy_outputs = recv_queue.get_nowait()
            # assigne outputs to linked env
            policy_outputs = process_policy_outputs(policy_outputs)
            next_rets = self.env.step(policy_outputs["action"])
            next_rets = process_env_rets(next_rets)
            # collect all rets and save as buffer
            episodes.record(rets, policy_outputs, next_rets)
            rets = next_rets

        recv_queue.shutdown(force=True)
        send_queue.shutdown(force=True)

        rollout_info = self.env.collect_info()
        if task_type == "rollout":
            episodes = list(
                episodes.to_numpy(
                    self.batch_mode, filter=list(desc["behavior_policies"].keys())
                ).values()
            )
            episodes = postprocessing(episodes)
            buffer_desc.batch_size = (
                self.env.batched_step_cnt
                if self.batch_mode == "time_step"
                else len(episodes)
            )

            indices = None
            while indices is None:
                batch = ray.get(
                    self.dataset_server.get_producer_index.remote(buffer_desc)
                )
                indices = batch.data

            buffer_desc.data = episodes
            buffer_desc.indices = indices
            self.dataset_server.save.remote(buffer_desc)

        ph = list(rollout_info.values())

        holder = {}
        for history, ds, k, vs in iter_many_dicts_recursively(*ph, history=[]):
            arr = [np.sum(_vs) for _vs in vs]
            prefix = "/".join(history)
            holder[prefix] = arr

        return {"total_fragment_length": self.env.batched_step_cnt, "eval_info": holder}
