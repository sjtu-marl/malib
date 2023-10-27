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

from typing import Dict, Any, List, Callable, Sequence, Tuple, Set
from abc import abstractmethod
from collections import defaultdict

import os
import time
import traceback
import logging
import pprint

import ray
import gym
import numpy as np

from ray.util import ActorPool
from torch.utils import tensorboard

from malib import settings
from malib.utils.typing import AgentID, BehaviorMode
from malib.utils.logging import Logger
from malib.utils.stopping_conditions import get_stopper
from malib.utils.monitor import write_to_tensorboard
from malib.common.strategy_spec import StrategySpec
from malib.common.task import RolloutTask
from malib.remote.interface import RemoteInterface
from malib.rollout.inference.client import InferenceClient
from malib.rollout.inference.env_runner import EnvRunner


PARAMETER_GET_TIMEOUT = 3
MAX_PARAMETER_GET_RETRIES = 10
DEFAULT_RESOURCE_CONFIG = dict(
    inference_server=dict(num_cpus=1, num_gpus=0),
    inference_client=dict(num_cpus=0, num_gpus=0),
)


logger = logging.getLogger(__name__)


def parse_rollout_info(raw_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of rollout information here.

    Args:
        raw_statistics (List[Dict[str, Any]]): A list of dict, each element is a result dict. Keys include
        - total_timesteps
        - FPS
        - evaluation

    Returns:
        Dict[str, Any]: A merged dict.
    """

    results = {"total_timesteps": 0, "FPS": 0.0}
    evaluation = []

    for e in raw_statistics:
        # when task mode is `simualtion` or `evaluation`, then
        #   evaluation result is not empty.
        if "evaluation" in e:
            evaluation.extend(e["evaluation"])

        for k, v in e.items():
            if k == "total_timesteps":
                results[k] += v
            elif k == "FPS":
                results[k] += v
            # else:
            #     raise ValueError(f"Unknow key: {k} / {v}")

    if len(evaluation) > 0:
        raw_eval_results = defaultdict(lambda: [])
        for e in evaluation:
            for k, v in e.items():
                if isinstance(v, (Tuple, List)):
                    v = sum(v)
                raw_eval_results[k].append(v)
        eval_results = {}
        for k, v in raw_eval_results.items():
            # convert v to array
            eval_results.update(
                {f"{k}_max": np.max(v), f"{k}_min": np.min(v), f"{k}_mean": np.mean(v)}
            )
        results["evaluation"] = eval_results
    return results


def log(message: str):
    logger.log(settings.LOG_LEVEL, f"(rollout worker) {message}")


def default_rollout_callback(coordinator: ray.ObjectRef, results: Dict[str, Any]):
    pass


def default_simulate_callback(coordinator: ray.ObjectRef, results: Dict[str, Any]):
    pass


def validate_runtime_configs(configs: Dict[str, Any]):
    """Validate runtime configuration.

    Args:
        configs (Dict[str, Any]): Raw runtime configuration

    Raises:
        AssertionError: Key not in configs
    """

    assert "fragment_length" in configs
    assert "max_step" in configs
    assert "num_eval_episodes" in configs
    assert "num_threads" in configs
    assert "num_env_per_thread" in configs
    assert "num_eval_threads" in configs
    assert "use_subproc_env" in configs
    assert "batch_mode" in configs
    assert "postprocessor_types" in configs
    assert "eval_interval" in configs


class RolloutWorker(RemoteInterface):
    def __init__(
        self,
        experiment_tag: str,
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable,
        agent_groups: Dict[str, Set],
        rollout_config: Dict[str, Any],
        log_dir: str,
        rollout_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        simulate_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        """Construct a rollout worker, consuming rollout/evaluation tasks.

        Args:
            env_desc (Dict[str, Any]): The environment description.
            agent_mapping_func (Callable): The agent mapping function, maps environment agents to runtime ids. \
                It is shared among all workers.
            rollout_config (Dict[str, Any]): Basic runtime configuration to control the rollout. Keys including
            * `fragment_length`: int, how many steps for each data collection and broadcasting.
            * `max_step`: int, the maximum step of each episode.
            * `num_eval_episodes`: int, the number of epsiodes for each evaluation.
            log_dir (str): Log directory.
            experiment_tag (str): Experiment tag, to create a data table.
            rollout_callback (Callable[[ray.ObjectRef, Dict[str, Any]], Any], optional): Callback function for rollout task, users can determine how \
                to cordinate with coordinator here. Defaults by None, indicating no coordination.
            simulate_callback (Callable[[ray.ObjectRef, Dict[str, Any]], Any]): Callback function for simulation task, users can determine \
                how to coordinate with coordinator here. Defaults by None, indicating no coordination.
            resource_config (Dict[str, Any], optional): Computional resource configuration, if not be specified, will load default configuraiton. Defaults to None.
            verbose (bool, optional): Enable logging or not. Defaults to True.
        """

        self.worker_indentifier = f"rolloutworker_{os.getpid()}"

        # map agents
        resource_config = resource_config or DEFAULT_RESOURCE_CONFIG

        self.env_description = env_desc
        self.env_agents = env_desc["possible_agents"]
        self.runtime_agent_ids = list(agent_groups.keys())
        self.agent_groups = agent_groups
        self.rollout_config: Dict[str, Any] = rollout_config

        validate_runtime_configs(self.rollout_config)

        self.inference_client_cls = InferenceClient.as_remote(
            **resource_config["inference_client"]
        )
        self.env_runner_cls = EnvRunner.as_remote(
            **resource_config["inference_server"]
        ).options(max_concurrency=100)

        self.env_runner_pool: ActorPool = self.init_env_runner_pool(
            env_desc, rollout_config, agent_mapping_func
        )
        self.inference_clients: Dict[
            AgentID, ray.ObjectRef
        ] = self.create_inference_clients()

        self.log_dir = log_dir
        self.rollout_callback = rollout_callback or default_rollout_callback
        self.simulate_callback = simulate_callback or default_simulate_callback
        self.tb_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self.experiment_tag = experiment_tag
        self.verbose = verbose

    def create_inference_clients(self) -> Dict[AgentID, ray.ObjectRef]:
        raise NotImplementedError

    def init_env_runner_pool(
        self,
        env_desc: Dict[str, Any],
        rollout_config: Dict[str, Any],
        agent_mapping_func: Callable,
    ) -> ActorPool:
        """Initialize an actor pool for the management of simulation tasks. Note the size of the \
            generated actor pool is determined by `num_threads + num_eval_threads`.

        Args:
            env_desc (Dict[str, Any]): Environment description.
            rollout_config (Dict[str, Any]): Runtime configuration, the given keys in this configuration \
                include:
                - `num_threads`: int, determines the size of this actor pool.
                - `num_env_per_thread`: int, indicates how many environments will be created for each thread.
                - `num_eval_threads`: int, determines how many threads will be created for the evaluation along the rollouts.
            agent_mapping_func (Callable): Agent mapping function which maps environment agents \
                to runtime ids, shared among all workers.

        Returns:
            ActorPool: An instance of `ActorPool`.
        """

        num_threads = rollout_config["num_threads"]
        num_env_per_thread = rollout_config["num_env_per_thread"]
        num_eval_threads = rollout_config["num_eval_threads"]

        env_runner_pool = ActorPool(
            [
                self.env_runner_cls.remote(
                    env_desc,
                    max_env_num=num_env_per_thread,
                    agent_groups=self.agent_groups,
                    use_subproc_env=rollout_config["use_subproc_env"],
                    batch_mode=rollout_config["batch_mode"],
                    postprocessor_types=rollout_config["postprocessor_types"],
                    training_agent_mapping=agent_mapping_func,
                )
                for _ in range(num_threads + num_eval_threads)
            ]
        )
        return env_runner_pool

    def rollout(self, task: RolloutTask):
        """Rollout, collecting training data when `data_entrypoints` is given, until meets the stopping conditions. The `active_agents` should be None or a none-empty list to specify active agents if rollout is not serve for evaluation.

        NOTE: the data collection will be triggered only for active agents.

        Args:
            task: None
        """

        stopper = get_stopper(task.stopping_conditions)
        active_agents = active_agents or self.env_agents
        runtime_strategy_specs = task.strategy_specs
        data_entrypoint_mapping = task.data_entrypoint_mapping

        rollout_config = self.rollout_config.copy()
        rollout_config.update(
            {
                "flag": "rollout",
                "strategy_specs": runtime_strategy_specs,
                "behavior_mode": BehaviorMode.EXPLORATION,
            }
        )
        total_timesteps = 0
        eval_results = {}
        epoch = 0
        performance = {
            "rollout_iter_rate": 0.0,
            "rollout_FPS": 0.0,
            "ave_rollout_FPS": 0.0,
        }

        self.set_running(True)

        start_time = time.time()
        while self.is_running():
            eval_step = (epoch + 1) % self.rollout_config["eval_interval"] == 0
            results = self.step_rollout(
                eval_step, rollout_config, data_entrypoint_mapping
            )
            total_timesteps += results["total_timesteps"]
            eval_results = results.get("evaluation", None)

            performance["rollout_iter_rate"] = (epoch + 1) / (time.time() - start_time)
            performance["rollout_FPS"] = results["FPS"]
            performance["ave_rollout_FPS"] = (
                performance["ave_rollout_FPS"] * epoch + results["FPS"]
            ) / (epoch + 1)

            if eval_results is not None:
                if self.verbose:
                    eval_results["performance"] = performance
                    formatted_results = pprint.pformat(eval_results)
                    Logger.info(f"Evaluation at epoch: {epoch}\n{formatted_results}")
                write_to_tensorboard(
                    self.tb_writer,
                    eval_results,
                    global_step=total_timesteps,
                    prefix="Evaluation",
                )

            write_to_tensorboard(
                self.tb_writer, performance, global_step=epoch, prefix="Performance"
            )

            # once call should stop, iteration += 1
            if stopper.should_stop(results):
                break
            epoch += 1

        self.rollout_callback(self.coordinator, results)
        return results

    @abstractmethod
    def step_rollout(
        self,
        eval_step: bool,
        rollout_config: Dict[str, Any],
        data_entrypoint_mapping: Dict[AgentID, str],
    ) -> List[Dict[str, Any]]:
        """The logic function to run rollout. Users must implment this method.

        Args:
            eval_step (bool): Indicate evaluation or not.
            rollout_config (Dict[str, Any]): Runtime configurations to control the amount of sampled data. Keys include:
            - `flag`: indicate the task type, the value is rollout.
            - `max_step`: indicates the maximum length of an episode.
            - `num_episodes`: indicates how many episodes will be collected.
            - `policy_distribution`: a dict describes the policy distribution.
            - `parameter_desc_dict`: a dict describes the parameter description.
            - `trainable_pairs`: a dict describes the trainable policy configuration, it is a mapping from `runtime_ids` \
                to a tuple of policy id and policy configuration.
            - `behavior_policies`: a dict maps runtime agents to policy ids, it specifies the behavior policy for available agents, \
                could be a subset of the full agent set.
            - `agent_group`: a dict that maps runtime agents to a list of environment agents, which describes the envrionment agents \
                governed by what runtime agent interface.
            - `fragment_length`: the maximum of collected data frames.
            data_entrypoint_mapping: ...

        Raises:
            NotImplementedError: _description_

        Returns:
            List[Dict[str, Any]]: Evaluation results, could be empty.
        """

    def assign_episode_id(self):
        return f"eps-{self.worker_indentifier}-{time.time()}"

    def close(self):
        """Terminate worker"""
        # TODO(ming): shut down actor pool
        # ray.kill(self.actor_pool)
        pass
