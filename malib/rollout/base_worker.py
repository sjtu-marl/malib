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

import os
import copy
import time
import traceback
import operator
import numpy as np

import ray

from functools import reduce
from collections import defaultdict
from ray.util import ActorPool

from malib import settings
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.typing import (
    AgentID,
    BufferDescription,
    TaskDescription,
    TaskRequest,
    TaskType,
    RolloutFeedback,
    Status,
    Sequence,
    Dict,
    Any,
    List,
    Callable,
)

from malib.utils.logger import Logger, get_logger
from malib.utils.stoppers import get_stopper
from malib.remote_interface import RemoteInterFace
from malib.rollout.inference_server import InferenceWorkerSet
from malib.rollout.inference_client import InferenceClient

PARAMETER_GET_TIMEOUT = 3
MAX_PARAMETER_GET_RETRIES = 10


def _parse_rollout_info(raw_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    holder = {}
    for history, ds, k, vs in iter_many_dicts_recursively(*raw_statistics, history=[]):
        prefix = "/".join(history)
        vs = reduce(operator.add, vs)
        holder[f"{prefix}_mean"] = np.mean(vs)
        holder[f"{prefix}_max"] = np.max(vs)
        holder[f"{prefix}_min"] = np.min(vs)
    return holder


class BaseRolloutWorker(RemoteInterFace):
    def __init__(
        self,
        worker_index: Any,
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable,
        runtime_configs: Dict[str, Any],
        experiment_config: Dict[str, Any],
    ):
        """Create a instance for simulations, rollout and evaluation. This base class initializes \
            all necessary servers and workers for rollouts. Including remote agent interfaces, \
                workers for simultaions.

        :param worker_index: The assigned worker index.
        :type worker_index: Any
        :param env_desc: The environment description.
        :type env_desc: Dict[str, Any]
        :param agent_mapping_func: The agent mapping function, maps environment agents to runtime ids. \
            It is shared among all workers.
        :type agent_mapping_func: Callable
        :param runtime_configs: The runtim configuraiton for the initialization of all workers.
        :type runtime_configs: Dict[str, Any]
        :param experiment_config: The experiment configuration, for the logging server connection.
        :type experiment_config: Dict[str, Any]
        """

        self._worker_index = worker_index
        self._env_description = env_desc
        self.global_step = 0

        self._coordinator = None
        self._parameter_server = None
        self._offline_dataset = None
        self._agents = env_desc["possible_agents"]

        # map agents
        agent_group = defaultdict(lambda: [])
        runtime_agent_ids = []
        for agent in env_desc["possible_agents"]:
            runtime_id = agent_mapping_func(agent)
            agent_group[runtime_id].append(agent)
            runtime_agent_ids.append(runtime_id)
        runtime_agent_ids = set(runtime_agent_ids)
        agent_group = dict(agent_group)

        self.runtime_agent_ids = runtime_agent_ids
        self.agent_group = agent_group
        self.worker_runtime_configs = runtime_configs

        self.init_servers()
        self.agent_interfaces = self.init_agent_interfaces(env_desc, runtime_agent_ids)
        self.actor_pool = self.init_actor_pool(
            env_desc, runtime_configs, agent_mapping_func
        )
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name="rollout_worker_{}".format(os.getpid()),
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **experiment_config,
        )

    def init_agent_interfaces(
        self, env_desc: Dict[str, Any], runtime_ids: Sequence[AgentID]
    ) -> Dict[AgentID, Any]:
        """Initialize agent interfaces which is a dict of `InterfaceWorkerSet`. The keys in the \
            dict is generated from the given agent mapping function.

        :param env_desc: Environment description.
        :type env_desc: Dict[str, Any]
        :param runtime_ids: Available runtime ids, generated with agent mapping function.
        :type runtime_ids: Sequence[AgentID]
        :return: A dict of agent interface.
        :rtype: Dict[AgentID, Any]
        """

        # interact with environment
        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]

        runtime_obs_spaces = {}
        runtime_act_spaces = {}

        for rid, agents in self.agent_group.items():
            runtime_obs_spaces[rid] = obs_spaces[agents[0]]
            runtime_act_spaces[rid] = act_spaces[agents[0]]

        agent_interfaces = {
            runtime_id: InferenceWorkerSet.remote(
                agent_id=runtime_id,
                observation_space=runtime_obs_spaces[runtime_id],
                action_space=runtime_act_spaces[runtime_id],
                parameter_server=self._parameter_server,
                governed_agents=self.agent_group[runtime_id],
            )
            for runtime_id in runtime_ids
        }

        return agent_interfaces

    def init_actor_pool(
        self,
        env_desc: Dict[str, Any],
        runtime_configs: Dict[str, Any],
        agent_mapping_func: Callable,
    ) -> ActorPool:
        """Initialize an actor pool for the management of simulation tasks. Note the size of the \
            generated actor pool is determined by `num_threads + num_eval_threads`.

        :param env_desc: Environment description.
        :type env_desc: Dict[str, Any]
        :param runtime_configs: Runtime configuraitons, the given keys in this configuration include
            - `num_threads`: determines the size of this actor pool.
            - `num_env_per_thread`: indicates how many environments will be created for each thread.
            - `num_eval_threads`: determines how many threads will be created for the evaluation along the rollouts.

        :type runtime_configs: Dict[str, Any]
        :param agent_mapping_func: Agent mapping function which maps environment agents to runtime ids, shared \
            among all workers.
        :type agent_mapping_func: Callable
        :return: An instance of `ActorPool`.
        :rtype: ActorPool
        """

        num_threads = runtime_configs["num_threads"]
        num_env_per_thread = runtime_configs["num_env_per_thread"]
        num_eval_threads = runtime_configs["num_eval_threads"]

        actor_pool = ActorPool(
            [
                InferenceClient.remote(
                    env_desc,
                    ray.get_actor(settings.OFFLINE_DATASET_ACTOR),
                    max_env_num=num_env_per_thread,
                    use_subproc_env=runtime_configs["use_subproc_env"],
                    batch_mode=runtime_configs["batch_mode"],
                    training_agent_mapping=agent_mapping_func,
                    postprocessor_types=runtime_configs["postprocessor_types"],
                )
                for _ in range(num_threads + num_eval_threads)
            ]
        )
        return actor_pool

    def init_servers(self):
        """Connect to coordinator and data servers here.

        :raises RuntimeError: Reached maximum retries.
        """

        retries = 100
        while True:
            try:
                if self._coordinator is None:
                    self._coordinator = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)

                if self._parameter_server is None:
                    self._parameter_server = ray.get_actor(
                        settings.PARAMETER_SERVER_ACTOR
                    )

                if self._offline_dataset is None:
                    self._offline_dataset = ray.get_actor(
                        settings.OFFLINE_DATASET_ACTOR
                    )
                self._status = Status.IDLE
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    self.logger.error("reached maximum retries")
                    raise RuntimeError(traceback.format_exc())
                else:
                    self.logger.warning(
                        f"waiting for coordinator server initialization ... {self._worker_index}\n{traceback.format_exc()}"
                    )
                    time.sleep(1)

    def get_status(self):
        return self._status

    def set_status(self, status):
        if status == self._status:
            return Status.FAILED
        else:
            self._status = status
            return Status.SUCCESS

    def rollout(self, task_desc: TaskDescription):
        """Collect training data asynchronously and stop it until the evaluation results meet the stopping conditions"""

        stopper = get_stopper(task_desc.content.stopper)(
            config=task_desc.content.stopper_config, tasks=None
        )
        merged_statics = {}
        epoch = 0
        print_every = 100

        # create data table
        trainable_pairs = task_desc.content.agent_involve_info.trainable_pairs
        # map trainable to runtime
        buffer_desc = {}
        for rid in self.runtime_agent_ids:
            pids = [
                trainable_pairs[aid][0]
                for aid in self.agent_group[rid]
                if aid in trainable_pairs
            ]
            if len(pids) == 0:
                continue
            buffer_desc[rid] = BufferDescription(
                env_id=self._env_description["config"][
                    "env_id"
                ],  # TODO(ziyu): this should be move outside "config"
                agent_id=self.agent_group[rid],
                policy_id=pids,
                capacity=None,
                sample_start_size=None,
            )
        for v in buffer_desc.values():
            ray.get(self._offline_dataset.create_table.remote(v))

        runtime_configs_template = self.worker_runtime_configs.copy()
        behavior_policy_mapping = {k: v[0] for k, v in trainable_pairs.items()}
        runtime_configs_template.update(
            {
                "flag": "rollout",
                "max_step": task_desc.content.max_step,
                "policy_distribution": task_desc.content.policy_distribution,
                "parameter_desc_dict": task_desc.content.agent_involve_info.meta_parameter_desc_dict,
                "trainable_pairs": trainable_pairs,
                "behavior_policies": behavior_policy_mapping,
                "agent_group": self.agent_group,
                "fragment_length": task_desc.content.fragment_length,
            }
        )
        start = time.time()
        total_num_frames = 0
        while not stopper(merged_statics, global_step=epoch):
            epoch_start = time.time()
            holder, epoch_num_frames = self.step_rollout(
                epoch, task_desc, buffer_desc, runtime_configs_template
            )
            total_num_frames += epoch_num_frames
            epoch_end = time.time()

            if (epoch + 1) % print_every == 0:
                Logger.info("\tepoch: %s (evaluation) %s", epoch, holder)
            if self.logger.is_remote:
                for k, v in holder.items():
                    self.logger.send_scalar(
                        tag="Evaluation/{}".format(k),
                        content=v,
                        global_step=epoch,
                    )
                self.logger.send_scalar(
                    tag="Performance/rollout_FPS",
                    content=epoch_num_frames / (epoch_end - epoch_start),
                    global_step=epoch,
                )
                self.logger.send_scalar(
                    tag="Performance/ave_rollout_FPS",
                    content=total_num_frames / (epoch_end - start),
                    global_step=epoch,
                )
            epoch += 1

        rollout_feedback = RolloutFeedback(
            worker_idx=self._worker_index,
            agent_involve_info=task_desc.content.agent_involve_info,
            statistics=holder,
        )
        self.callback(
            Status.NORMAL, task_desc, rollout_feedback, role="rollout", relieve=True
        )

    def simulate(self, task_desc: TaskDescription):
        """Handling simulation task."""

        combinations = task_desc.content.policy_combinations
        agent_involve_info = task_desc.content.agent_involve_info
        raw_statistics = self.step_simulation(task_desc)

        for statistics, combination in zip(raw_statistics, combinations):
            holder = _parse_rollout_info([statistics])
            rollout_feedback = RolloutFeedback(
                worker_idx=self._worker_index,
                agent_involve_info=agent_involve_info,
                statistics=holder,
                policy_combination={k: p for k, (p, _) in combination.items()},
            )
            task_req = TaskRequest.from_task_desc(
                task_desc=task_desc,
                task_type=TaskType.UPDATE_PAYOFFTABLE,
                content=rollout_feedback,
            )
            self._coordinator.request.remote(task_req)
        self.set_status(Status.IDLE)

    def step_rollout(
        self,
        n_step: int,
        task_desc: TaskDescription,
        buffer_desc: BufferDescription,
        runtime_configs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """The logic function to run rollout. Users must implment this method.

        :param n_step: Indicates the rollout iteration.
        :type n_step: int
        :param task_desc: The instance of task description.
        :type task_desc: TaskDescription
        :param buffer_desc: The instance of buffer description
        :type buffer_desc: BufferDescription
        :param runtime_configs: Runtime configurations to control the amount of sampled data. Keys include:
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

        :type runtime_configs: Dict[str, Any]
        :raises NotImplementedError: Not implemented error
        :return: A list of dict which logs the rollout information.
        :rtype: List[Dict[str, Any]]
        """
        raise NotImplementedError

    def step_simulation(self, task_desc: TaskDescription) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def callback(
        self,
        status: Status,
        task_desc: TaskDescription,
        content: Any,
        role: str,
        relieve: bool,
    ):
        if role == "simulation":
            task_req = TaskRequest.from_task_desc(
                task_desc=task_desc,
                task_type=TaskType.UPDATE_PAYOFFTABLE,
                content=content,
            )
            self._coordinator.request.remote(task_req)
        else:
            if status is not Status.LOCKED:
                parameter_desc_dict = (
                    content.agent_involve_info.meta_parameter_desc_dict
                )
                for agent, (
                    pid,
                    _,
                ) in content.agent_involve_info.trainable_pairs.items():
                    parameter_desc = copy.copy(
                        parameter_desc_dict[agent].parameter_desc_dict[pid]
                    )
                    parameter_desc.type = "parameter"
                    parameter_desc.lock = True
                    parameter_desc.data = (
                        self.agent_interfaces[agent].policies[pid].state_dict()
                    )
                    _ = ray.get(self._parameter_server.push.remote(parameter_desc))
                self._coordinator.request.remote(
                    TaskRequest.from_task_desc(
                        task_desc=task_desc,
                        task_type=TaskType.EVALUATE,
                        content=content,
                    )
                )
        if relieve:
            # unlock worker
            self.set_status(Status.IDLE)

    def assign_episode_id(self):
        return f"eps-{self._worker_index}-{time.time()}"

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.

        :param policy_distribution: Dict[AgentID, Dict[PolicyID, float]], default by None
        """

        for aid, interface in self.agent_interfaces.items():
            if policy_distribution is None or aid not in policy_distribution:
                pass
            else:
                interface.set_behavior_dist(policy_distribution[aid])

    def save_model(self):
        """Save policy model to log directory."""

        save_dir = os.path.join(
            settings.LOG_DIR,
            self._kwargs["exp_cfg"]["expr_group"],
            self._kwargs["exp_cfg"]["expr_name"],
            "models",
        )
        for aid, interface in self._agent_interfaces.items():
            _save_dir = os.path.join(save_dir, aid)
            interface.save(_save_dir)

    def close(self):
        """Terminate worker"""

        # TODO(ming): store worker's state
        self.logger.info(f"Worker: {self._worker_index} has been terminated.")
        for agent in self.agent_interfaces.values():
            agent.close()
