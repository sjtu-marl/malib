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

from typing import Dict, Any, List, Callable, Sequence, Tuple, Set, Union
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
from malib.rollout.config import RolloutConfig
from malib.rollout.inference.client import InferenceClient
from malib.rollout.inference.env_runner import BasicEnvRunner


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

    return raw_statistics


def log(message: str):
    logger.log(settings.LOG_LEVEL, f"(rollout worker) {message}")


def default_rollout_callback(results: Dict[str, Any]):
    pass


def default_simulate_callback(results: Dict[str, Any]):
    pass


class RolloutWorker(RemoteInterface):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        agent_groups: Dict[str, Tuple],
        rollout_config: Union[RolloutConfig, Dict[str, Any]],
        log_dir: str,
        rollout_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        simulate_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        """Construct a rollout worker, consuming rollout/evaluation tasks.

        Args:
            env_desc (Dict[str, Any]): The environment description.
            rollout_config (Dict[str, Any]): Basic runtime configuration to control the rollout. Keys including
            * `fragment_length`: int, how many steps for each data collection and broadcasting.
            * `max_step`: int, the maximum step of each episode.
            * `num_eval_episodes`: int, the number of epsiodes for each evaluation.
            log_dir (str): Log directory.
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
        self.rollout_config = RolloutConfig.from_raw(rollout_config)

        # create environment runner, handling evaluation or rollout task
        env_runner_resource_config = resource_config["inference_server"]
        self.env_runner = self.create_env_runner(
            env_desc, env_runner_resource_config, self.rollout_config
        )

        self.log_dir = log_dir
        self.rollout_callback = rollout_callback or default_rollout_callback
        self.simulate_callback = simulate_callback or default_simulate_callback
        self.tb_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self.verbose = verbose

    def create_env_runner(
        self,
        env_desc: Dict[str, Any],
        resource_config: Dict[str, Any],
        rollout_config: RolloutConfig,
    ) -> ActorPool:
        """Initialize an actor pool for the management of simulation tasks. Note the size of the \
            generated actor pool is determined by `num_threads + num_eval_threads`.

        Args:
            env_desc (Dict[str, Any]): Environment description.
            rollout_config (RolloutConfig): Rollout configuration.
            agent_mapping_func (Callable): Agent mapping function which maps environment agents \
                to runtime ids, shared among all workers.

        Returns:
            ActorPool: An instance of `ActorPool`.
        """

        env_runner_cls = BasicEnvRunner.as_remote(**resource_config)
        env_runner = env_runner_cls.remote(
            env_func=lambda: env_desc["creator"](**env_desc["config"]),
            max_env_num=rollout_config.n_envs_per_worker,
            use_subproc_env=rollout_config.use_subproc_env,
            agent_groups=self.agent_groups,
            inference_entry_points=rollout_config.inference_entry_points,
        )

        return env_runner

    def rollout(self, task: RolloutTask):
        """Rollout, collecting training data when `data_entrypoints` is given, until meets the stopping conditions. The `active_agents` should be None or a none-empty list to specify active agents if rollout is not serve for evaluation.

        NOTE: the data collection will be triggered only for active agents.

        Args:
            task: None
        """

        stopper = get_stopper(task.stopping_conditions)

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
            eval_step = (epoch + 1) % self.rollout_config.eval_interval == 0
            results = self.step_rollout(
                eval_step,
                task.strategy_specs,
                task.data_entrypoints,
            )
            total_timesteps += results["total_timesteps"]

            # performance["rollout_iter_rate"] = (epoch + 1) / (time.time() - start_time)
            # performance["rollout_FPS"] = results["FPS"]
            # performance["ave_rollout_FPS"] = (
            #     performance["ave_rollout_FPS"] * epoch + results["FPS"]
            # ) / (epoch + 1)

            # if self.verbose:
            #     eval_results["performance"] = performance
            #     formatted_results = pprint.pformat(eval_results)
            #     Logger.info(f"Evaluation at epoch: {epoch}\n{formatted_results}")

            write_to_tensorboard(
                self.tb_writer,
                results,
                global_step=total_timesteps,
                prefix="Rollouts",
            )

            write_to_tensorboard(
                self.tb_writer, performance, global_step=epoch, prefix="Performance"
            )

            # once call should stop, iteration += 1
            if stopper.should_stop(results):
                break
            epoch += 1

        self.rollout_callback(results)

        return results

    @abstractmethod
    def step_rollout(
        self,
        eval_step: bool,
        strategy_specs: Dict[AgentID, StrategySpec],
        data_entrypoints: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """The logic function to run rollout. Users must implment this method.

        Args:
            eval_step (bool): Indicate evaluation or not.
            data_entrypoints: ...

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
