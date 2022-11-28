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

from typing import Dict, Any, List, Callable, Sequence, Tuple
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
from ray.util.queue import Queue
from torch.utils import tensorboard

from malib import settings
from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.utils.stopping_conditions import get_stopper
from malib.utils.monitor import write_to_tensorboard
from malib.common.strategy_spec import StrategySpec
from malib.remote.interface import RemoteInterface
from malib.rollout.inference.ray.server import (
    RayInferenceWorkerSet as RayInferenceServer,
)
from malib.rollout.inference.ray.client import RayInferenceClient


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


def validate_agent_group(
    agent_group: Dict[str, List[AgentID]],
    full_keys: List[AgentID],
    observation_spaces: Dict[AgentID, gym.Space],
    action_spaces: Dict[AgentID, gym.Space],
) -> None:
    """Validate agent group, check spaces.

    Args:
        agent_group (Dict[str, List[AgentID]]): A dict, mapping from runtime ids to lists of agent ids.
        full_keys (List[AgentID]): A list of original environment agent ids.
        observation_spaces (Dict[AgentID, gym.Space]): Agent observation space dict.
        action_spaces (Dict[AgentID, gym.Space]): Agent action space dict.

    Raises:
        RuntimeError: Agents in a same group should share the same observation space and action space.
        NotImplementedError: _description_
    """
    for agents in agent_group.values():
        select_obs_space = observation_spaces[agents[0]]
        select_act_space = action_spaces[agents[0]]
        for agent in agents[1:]:
            assert type(select_obs_space) == type(observation_spaces[agent])
            assert select_obs_space.shape == observation_spaces[agent].shape
            assert type(select_act_space) == type(action_spaces[agent])
            assert select_act_space.shape == action_spaces[agent].shape


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
        rollout_config: Dict[str, Any],
        log_dir: str,
        rollout_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        simulate_callback: Callable[[ray.ObjectRef, Dict[str, Any]], Any] = None,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        """Create a instance for simulations, rollout and evaluation. This base class initializes \
            all necessary servers and workers for rollouts. Including remote agent interfaces, \
                workers for simultaions.

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
        agent_group = defaultdict(lambda: [])
        runtime_agent_ids = []
        for agent in env_desc["possible_agents"]:
            runtime_id = agent_mapping_func(agent)
            agent_group[runtime_id].append(agent)
            runtime_agent_ids.append(runtime_id)
        runtime_agent_ids = set(runtime_agent_ids)
        agent_group = dict(agent_group)
        resource_config = resource_config or DEFAULT_RESOURCE_CONFIG

        # valid agent group
        validate_agent_group(
            agent_group=agent_group,
            full_keys=env_desc["possible_agents"],
            observation_spaces=env_desc["observation_spaces"],
            action_spaces=env_desc["action_spaces"],
        )

        self.env_description = env_desc
        self.env_agents = env_desc["possible_agents"]
        self.runtime_agent_ids = runtime_agent_ids
        self.agent_group = agent_group
        self.rollout_config: Dict[str, Any] = rollout_config

        validate_runtime_configs(self.rollout_config)

        self.coordinator = None
        self.dataset_server = None
        self.parameter_server = None

        self.init_servers()

        if rollout_config["inference_server"] == "local":
            self.inference_server_cls = None
            self.inference_client_cls = RayInferenceClient.as_remote(
                **resource_config["inference_client"]
            )
        elif rollout_config["inference_server"] == "ray":
            self.inference_client_cls = RayInferenceClient.as_remote(
                **resource_config["inference_client"]
            )
            self.inference_server_cls = RayInferenceServer.as_remote(
                **resource_config["inference_server"]
            ).options(max_concurrency=100)

        else:
            raise ValueError(
                "unexpected inference server type: {}".format(
                    rollout_config["inference_server"]
                )
            )

        self.agent_interfaces = self.init_agent_interfaces(env_desc, runtime_agent_ids)
        self.actor_pool: ActorPool = self.init_actor_pool(
            env_desc, rollout_config, agent_mapping_func
        )

        self.log_dir = log_dir
        self.rollout_callback = rollout_callback or default_rollout_callback
        self.simulate_callback = simulate_callback or default_simulate_callback
        self.tb_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self.experiment_tag = experiment_tag
        self.verbose = verbose

    def init_agent_interfaces(
        self, env_desc: Dict[str, Any], runtime_ids: Sequence[AgentID]
    ) -> Dict[AgentID, Any]:
        """Initialize agent interfaces which is a dict of `InterfaceWorkerSet`. The keys in the \
            dict is generated from the given agent mapping function.

        Args:
            env_desc (Dict[str, Any]): Environment description.
            runtime_ids (Sequence[AgentID]): Available runtime ids, generated with agent mapping function.

        Returns:
            Dict[AgentID, Any]: A dict of `InferenceWorkerSet`, mapping from `runtime_ids` to `ray.ObjectRef(s)`
        """

        # interact with environment
        if self.inference_server_cls is None:
            return None

        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]

        runtime_obs_spaces = {}
        runtime_act_spaces = {}

        for rid, agents in self.agent_group.items():
            runtime_obs_spaces[rid] = obs_spaces[agents[0]]
            runtime_act_spaces[rid] = act_spaces[agents[0]]

        agent_interfaces = {
            runtime_id: self.inference_server_cls.remote(
                agent_id=runtime_id,
                observation_space=runtime_obs_spaces[runtime_id],
                action_space=runtime_act_spaces[runtime_id],
                parameter_server=self.parameter_server,
                governed_agents=self.agent_group[runtime_id],
            )
            for runtime_id in runtime_ids
        }

        return agent_interfaces

    def init_actor_pool(
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

        actor_pool = ActorPool(
            [
                self.inference_client_cls.remote(
                    env_desc,
                    ray.get_actor(settings.OFFLINE_DATASET_ACTOR),
                    max_env_num=num_env_per_thread,
                    use_subproc_env=rollout_config["use_subproc_env"],
                    batch_mode=rollout_config["batch_mode"],
                    postprocessor_types=rollout_config["postprocessor_types"],
                    training_agent_mapping=agent_mapping_func,
                )
                for _ in range(num_threads + num_eval_threads)
            ]
        )
        return actor_pool

    def init_servers(self):
        """Connect to data servers.

        Raises:
            RuntimeError: Runtime errors.
        """

        retries = 100
        while True:
            try:
                if self.parameter_server is None:
                    self.parameter_server = ray.get_actor(
                        settings.PARAMETER_SERVER_ACTOR
                    )

                if self.dataset_server is None:
                    self.dataset_server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise RuntimeError(traceback.format_exc())
                else:
                    logger.log(
                        logging.WARNING,
                        f"waiting for coordinator server initialization ... {self.worker_indentifier}",
                    )
                    time.sleep(1)

    def rollout(
        self,
        runtime_strategy_specs: Dict[str, StrategySpec],
        stopping_conditions: Dict[str, Any],
        data_entrypoints: Dict[str, str],
        trainable_agents: List[AgentID] = None,
    ):
        """Run rollout procedure, collect data until meets the stopping conditions.

        NOTE: the data collection will be triggered only for trainable agents.

        Args:
            runtime_strategy_specs (Dict[str, StrategySpec]): A dict of strategy spec, mapping from runtime id to `StrategySpec`.
            stopping_conditions (Dict[str, Any]): A dict of stopping conditions.
            data_entrypoints (Dict[str, str]): Mapping from runtimeids to dataentrypoint names.
            trainable_agents (List[AgentID], optional): A list of environment agent id. Defaults to None, which means all environment agents will be trainable.
        """

        stopper = get_stopper(stopping_conditions)
        trainable_agents = trainable_agents or self.env_agents
        queue_info_dict: Dict[str, Tuple[str, Queue]] = {
            rid: None for rid in self.runtime_agent_ids
        }
        for rid, identifier in data_entrypoints.items():
            queue_id, queue = ray.get(
                self.dataset_server.start_producer_pipe.remote(name=identifier)
            )
            queue_info_dict[rid] = (queue_id, queue)

        rollout_config = self.rollout_config.copy()
        rollout_config.update(
            {
                "flag": "rollout",
                "strategy_specs": runtime_strategy_specs,
                "trainable_agents": trainable_agents,
                "agent_group": self.agent_group,
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
        # TODO(ming): share the stopping conditions here
        while self.is_running():
            eval_step = (epoch + 1) % self.rollout_config["eval_interval"] == 0
            results = self.step_rollout(eval_step, rollout_config, queue_info_dict)
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

    def simulate(self, runtime_strategy_specs: Dict[str, StrategySpec]):
        """Handling simulation task."""

        runtime_config_template = self.rollout_config.copy()
        runtime_config_template.update(
            {
                "flag": "simulation",
            }
        )

        results: Dict[str, Any] = self.step_simulation(
            runtime_strategy_specs, runtime_config_template
        )

        self.simulate_callback(self.coordinator, results)
        return results

    @abstractmethod
    def step_rollout(
        self,
        eval_step: bool,
        rollout_config: Dict[str, Any],
        dataset_writer_info_dict: Dict[str, Any],
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

        Raises:
            NotImplementedError: _description_

        Returns:
            List[Dict[str, Any]]: Evaluation results, could be empty.
        """

    @abstractmethod
    def step_simulation(
        self,
        runtime_strategy_specs: Dict[str, StrategySpec],
        rollout_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Logic function for running simulation of a list of strategy spec dict.

        Args:
            runtime_strategy_specs (Dict[str, StrategySpec]): A strategy spec dict.
            rollout_config (Dict[str, Any]): Runtime configuration template.

        Raises:
            NotImplementedError: Not implemented error.

        Returns:
            Dict[str, Any]: A evaluation results.
        """

    def assign_episode_id(self):
        return f"eps-{self.worker_indentifier}-{time.time()}"

    def close(self):
        """Terminate worker"""
        # TODO(ming): shut down actor pool
        # ray.kill(self.actor_pool)
        pass
