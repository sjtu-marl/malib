"""
Basic class of agent interface. Users can implement their custom training workflow by inheriting this class.
"""

import copy
from dataclasses import dataclass
import threading
import time
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Dict, Any, Tuple, Callable, Union, Sequence

import gym
import numpy as np
import ray

from malib import settings
from malib.utils.stoppers import get_stopper
from malib.utils.typing import (
    PolicyID,
    ParameterDescription,
    TaskType,
    TaskDescription,
    Status,
    MetaParameterDescription,
    BufferDescription,
    TrainingFeedback,
    TaskRequest,
    AgentID,
    MetricEntry,
    DataTransferType,
)
from malib.utils import errors
from malib.utils.logger import get_logger, Log
from malib.algorithm.common.policy import Policy, TabularPolicy
from malib.algorithm.common.trainer import Trainer


AgentFeedback = namedtuple("AgentFeedback", "id, trainable_pairs, state_id, statistics")
AgentTaggedFeedback = namedtuple("AgentTaggedFeedback", "id, content")

AgentFeedback.__doc__ = """\
Policy adding feedback.
"""

AgentFeedback.id.__doc__ = """\
Dict[str, Any] - Training agent id.
"""

AgentFeedback.state_id.__doc__ = """\
ObjectRef - State id, linked to mutable object.
"""

AgentFeedback.trainable_pairs.__doc__ = """\
Dict[AgentID, Tuple[PolicyID, Any] - Mapping from environment agents to policy description tuple.
"""

AgentFeedback.statistics.__doc__ = """\
Dict[str, Any] - A dictionary of statistics.
"""

AgentTaggedFeedback.__doc__ = """\
Agent feedback wraps stationary contents tagged with environment agent id.
"""

AgentTaggedFeedback.id.__doc__ = """\
AgentID - Environment agent id.
"""

AgentTaggedFeedback.content.__doc__ = """\
Any - Stationary results
"""


class AgentInterface(metaclass=ABCMeta):
    """Base class of agent interface, for training"""

    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int,
        algorithm_mapping: Callable = None,
    ):
        """
        :param str assign_id: Specify the agent interface id.
        :param Dict[str,Any] env_desc: Environment description.
        :param Dict[str,Any] algorithm_candidates: A dict of feasible algorithms.
        :param Dict[AgentID,gym.spaces.Space] observation_spaces: A dict of raw environment observation spaces.
        :param Dict[AgentID,gym.spaces.Space] action_spaces: A dict of raw environment action spaces.
        :param Dict[str,Any] exp_cfg: Experiment description.
        :param int population_size: The maximum size of policy pool.
        :param Optional[Callable] algorithm_mapping: Mapping registered agents to algorithm candidates, optional
            default is None.
        """

        self._id = assign_id
        self._env_desc = env_desc
        self._algorithm_candidates = algorithm_candidates
        self._observation_spaces = observation_spaces
        self._action_spaces = action_spaces
        self._population_size = population_size
        self._policies = {}
        self._trainers = {}
        self._agent_to_pids = {}
        self._offline_dataset = None
        self._coordinator = None
        self._parameter_server = None
        self._parameter_desc: Dict[PolicyID, ParameterDescription] = {}
        self._meta_parameter_desc = {}
        self._algorithm_mapping_func = algorithm_mapping
        self._training_agent_mapping = training_agent_mapping
        self._group = []
        self._global_step = 0

        self._param_desc_lock = threading.Lock()
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name=f"training_agent_interface_{self._id}",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )

    def get_policies(self) -> Dict[PolicyID, Policy]:
        """Get a dict of policies.

        :return: A dict of policies.
        """

        return self._policies

    def agent_group(self) -> Tuple[AgentID]:
        """Return a tuple of registered environment agents.

        :return: A tuple of agent ids.
        """

        return self._group

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def register_env_agent(
        self, env_agent_id: Union[AgentID, Sequence[AgentID]]
    ) -> None:
        """Register environment agents.

        :param Union[AgentID,Sequence[AgentID]] env_agent_id: Environment agent id(s), could be an agent id or a list
            of it.
        :return: None
        """

        if isinstance(env_agent_id, AgentID):
            assert env_agent_id not in self._group, (env_agent_id, self._group)
            self._group.append(env_agent_id)
            self._agent_to_pids[env_agent_id] = []
        else:
            env_agent_ids = list(env_agent_id)
            for e in env_agent_ids:
                assert e not in self._group
                self._agent_to_pids[e] = []
            self._group.extend(env_agent_ids)

    @property
    def policies(self) -> Dict[PolicyID, Policy]:
        """Return a dict of policies.

        :return: {policy_id: policy}.
        """

        return self._policies

    @property
    def algorithm_candidates(self) -> Dict[str, Any]:
        """Return a dict of algorithm configurations supported in this interface, users can use one of them to create
        policy instance.

        :return: {algorithm_name: algorithm_configuration}
        """
        return self._algorithm_candidates

    def start(self) -> None:
        """Retrieve the handlers of coordinator server, parameter server and offline dataset server.

        Note:
            This method can only be called when remote servers, i.e. `Coordinator`, `Parameter` and `OfflineDataset`
            servers have been started.

        Example:
            >>> coordinator = CoordinatorServer.remote(...)
            >>> offline_dataset = OfflineDatasetServer.remote(...)
            >>> parameter_server = ParameterServer.remote(...)
            >>> # you can choose make it work as an actor or not
            >>> agent_interface = AgentInterface(...)
            >>> agent_interface.start()

        :return: None
        """

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
                self.logger.debug(f"agent={self._id} got coordinator handler")
                break
            except Exception as e:
                self.logger.debug(f"Waiting for coordinator server... {e}")
                time.sleep(1)
                continue

    def require_parameter_desc(self, state_id) -> Dict:
        """Return a meta parameter description.

        :param ObjectRef state_id: Ray object ref
        """

        with self._param_desc_lock:
            return self._meta_parameter_desc

    def get_stationary_state(self, state_id: ray.ObjectID) -> AgentTaggedFeedback:
        """Return stationary policy descriptions."""

        if state_id:
            res = {
                env_aid: [
                    (pid, self._policies[pid].description) for pid in state_id[env_aid]
                ]
                for env_aid in self._group
            }
        else:
            res = {
                env_aid: [
                    (pid, self._policies[pid].description)
                    for pid in self._agent_to_pids[env_aid]
                ]
                for env_aid in self._group
            }
        return AgentTaggedFeedback(self._id, content=res)

    def push(self, env_aid: AgentID, pid: PolicyID) -> Status:
        """Coordinate with remote parameter server, default behavior is to push parameters.

        :param AgentID env_aid: registered agent id
        :param PolicyID pid: registered policy id
        :return a TableStatus code.
        """

        parameter_desc = self._parameter_desc[pid]
        policy = self._policies[pid]
        parameter_desc.data = policy.state_dict()
        parameter_desc.version += 1
        status = ray.get(self._parameter_server.push.remote(parameter_desc))
        parameter_desc.data = None
        return status

    def pull(self, env_aid: AgentID, pid: PolicyID) -> Status:
        """Pull parameter from parameter server, default is None

        :param AgentID env_aid: Registered agent id.
        :param PolicyID pid: Registered policy id.
        :return: Status code.
        """

        return Status.NORMAL

    def request_data(
        self, buffer_desc: Union[BufferDescription, Dict[AgentID, BufferDescription]]
    ) -> Tuple[Any, str]:
        """Request training data from remote `OfflineDatasetServer`.

        Note:
            This method could only be called in multi-instance scenarios. Or, `OfflineDataset` and `CoordinatorServer`
            have been started.

        :param Dict[AgentID,BufferDescription] buffer_desc: A dictionary of agent buffer descriptions.
        :return: A tuple of agent batches and information.
        """

        if isinstance(buffer_desc, Dict):
            res = {}
            # multiple tasks
            tasks = [
                self._offline_dataset.sample.remote(v) for v in buffer_desc.values()
            ]
            while len(tasks) > 0:
                dones, tasks = ray.wait(tasks)
                for done in dones:
                    batch, info = ray.get(done)
                    if batch.data is None:
                        # push task
                        tasks.append(
                            self._offline_dataset.sample.remote(
                                buffer_desc[batch.identity]
                            )
                        )
                    else:
                        res[batch.identity] = batch.data
        else:
            res = None
            while True:
                batch, info = ray.get(self._offline_dataset.sample.remote(buffer_desc))
                if batch.data is None:
                    continue
                else:
                    res = batch.data
                    break
        return res, info

    def gen_buffer_description(
        self,
        agent_policy_mapping: Dict[AgentID, PolicyID],
        batch_size: int,
        sample_mode: str,
    ):
        """Generate buffer description.

        :param AgentID aid: Environment agent id.
        :param PolicyID pid: Policy id.
        :param int batch_size: Sample batch size.
        :param str sample_mode: sample mode
        :return: A buffer description entity.
        """

        return {
            aid: BufferDescription(
                env_id=self._env_desc["config"]["env_id"],
                agent_id=aid,
                policy_id=pid,
                batch_size=batch_size,
                sample_mode=sample_mode,
            )
            for aid, (pid, _) in agent_policy_mapping.items()
        }

    @Log.method_timer(enable=settings.PROFILING)
    def train(self, task_desc: TaskDescription, training_config: Dict[str, Any] = None):
        """Handling training task with a given task description.

        Note:
            This method could only be called in multi-instance scenarios. Or, `OfflineDataset` and `CoordinatorServer`
            have been started.

        :param TaskDescription task_desc: Task description entity, `task_desc.content` must be a `TrainingTask` entity.
        :param Dict[str,Any] training_config: Training configuration. Default to None.
        :return: None
        """

        training_task = task_desc.content
        agent_involve_info = training_task.agent_involve_info
        # retrieve policy ids required to training

        batch_size = training_config.get("batch_size", 64)
        sample_mode = training_task.mode

        buffer_desc = self.gen_buffer_description(
            agent_involve_info.trainable_pairs, batch_size, sample_mode
        )
        policy_id_mapping = {
            env_aid: pid
            for env_aid, (pid, _) in agent_involve_info.trainable_pairs.items()
        }

        self.logger.info(
            f"Start training task for interface={self._id} with policy mapping:\n\t{policy_id_mapping} -----"
        )
        # register sub tasks
        stopper = get_stopper(training_task.stopper)(
            tasks=[env_aid for env_aid in policy_id_mapping],
            config=training_task.stopper_config,
        )
        epoch = 0
        statistics = {}
        status = None

        # sync parameters if implemented
        for env_aid, pid in policy_id_mapping.items():
            self.pull(env_aid, pid)

        old_policy_id_mapping = copy.deepcopy(policy_id_mapping)

        while not stopper(statistics, global_step=epoch) and not stopper.all():
            # add timer: key to identify object, tag to log
            # FIXME(ming): key for logger has been discarded!
            with Log.timer(
                log=True,
                logger=self.logger,
                tag=f"time/TrainingInterface_{self._id}/data_request",
                global_step=epoch,
            ):
                start = time.time()
                batch, size = self.request_data(buffer_desc)

            with Log.stat_feedback(
                log=settings.STATISTIC_FEEDBACK,
                logger=self.logger,
                worker_idx=self._id,
                global_step=epoch,
                group="training",
            ) as (statistic_seq, processed_statistics):
                # a dict of dict of metric entry {agent: {item: MetricEntry}}
                statistics = self.optimize(policy_id_mapping, batch, training_config)
                statistic_seq.append(statistics)
                # NOTE(ming): if it meets the update interval, parameters will be pushed to remote parameter server
                # the returns `status` code will determine whether we should stop the training or continue it.
                if (epoch + 1) % training_config["update_interval"] == 0:
                    for env_aid in self._group:
                        pid = policy_id_mapping[env_aid]
                        status = self.push(env_aid, pid)
                        if status.locked:
                            # terminate sub task tagged with env_id
                            stopper.set_terminate(env_aid)
                            # and remove buffer request description
                            assert isinstance(buffer_desc, Dict)
                            buffer_desc.pop(env_aid)
                            # also training poilcy id mapping
                            policy_id_mapping.pop(env_aid)
                        else:
                            self.pull(env_aid, pid)
                epoch += 1
                self._global_step += 1
        print(f"**** training for mapping: {old_policy_id_mapping} finished")
        if status is not None and not status.locked:
            for aid, pid in old_policy_id_mapping.items():
                parameter_desc = copy.copy(self._parameter_desc[pid])
                parameter_desc.lock = True
                policy = self._policies[pid]
                parameter_desc.data = policy.state_dict()
                status = ray.get(self._parameter_server.push.remote(parameter_desc))
                assert status.locked, status

                # call evaluation request
            task_request = TaskRequest(
                task_type=TaskType.EVALUATE,
                content=TrainingFeedback(
                    agent_involve_info=training_task.agent_involve_info,
                    statistics=stopper.info,
                ),
            )
            self._coordinator.request.remote(task_request)

    def register_policy(self, pid: PolicyID, policy: Policy) -> None:
        """Register policy into policy pool.

        :param pid: PolicyID, policy id
        :param policy: Policy, a policy instance
        :return: None
        """

        assert pid not in self._policies
        self._policies[pid] = policy

    def parameter_desc_gen(
        self, env_aid: AgentID, policy_id: PolicyID, trainable: bool, data=None
    ):
        """Generate a parameter description entity. The returned description will not load policy weights by default.

        :param AgentID env_aid: Environment agent id.
        :param PolicyID policy_id: Policy id.
        :param bool trainable: Specify whether the policy is trainable or not.
        :param Any data: Parameter data. Default to None
        :return: A `ParameterDescription` entity related to policy tagged with `policy_id`.
        """

        return ParameterDescription(
            env_id=self._env_desc["config"]["env_id"],
            identify=env_aid,
            id=policy_id,
            time_stamp=time.time(),
            description=self._policies[policy_id].description,
            data=data,
            lock=not trainable,
        )

    def check_population_size(self) -> None:
        """Called before policy adding, to check whether there is enough space to add `len(self._group)` policies.

        :raise: errors.NoEnoughSpace
        :return: None
        """

        if self._population_size < 0:
            return
        if len(self.policies) + len(self._group) < self._population_size:
            return
        else:
            raise errors.NoEnoughSpace(
                f"No more space to create {len(self._group)} policies"
            )

    def add_policy(self, task_desc: TaskDescription):
        """Handling policy adding task with a given task description. This method will parse the transferred task
        description to create new policies for all environment agents registered in this interface (one agent one policy
        ). And then local parameters will be sent to remote parameter server, two task requests will be sent to the
        `CoordinatorServer`.

        Note:
            This method could only be called in multi-instance scenarios. Or, `OfflineDataset` and `CoordinatorServer`
            have been started.
        """

        # add policies for agents, the first one policy will be set to non-trainable
        if len(self.policies) < 1:
            trainable = False
        else:
            trainable = True

        self.check_population_size()
        policy_dict: Dict[AgentID, Tuple[PolicyID, Policy]] = {
            env_aid: self.add_policy_for_agent(env_aid, trainable)
            for env_aid in self._group
        }

        pending_tasks = []
        with self._param_desc_lock:
            for env_aid, (pid, policy) in policy_dict.items():
                self._agent_to_pids[env_aid].append(pid)
                parameter_desc = self.parameter_desc_gen(
                    env_aid, pid, trainable, data=policy.state_dict()
                )
                pending_tasks.append(
                    self._parameter_server.push.remote(parameter_desc=parameter_desc)
                )
                parameter_desc.data = None
                self._parameter_desc[pid] = parameter_desc
                if self._meta_parameter_desc.get(env_aid, None) is None:
                    self._meta_parameter_desc[env_aid] = MetaParameterDescription(
                        meta_pid=env_aid, parameter_desc_dict={}
                    )

                self._meta_parameter_desc[env_aid].parameter_desc_dict[
                    pid
                ] = self.parameter_desc_gen(env_aid, pid, trainable)

        # wait until all parameter push tasks ended
        # XXX(ming): push parameter in add_policy stage should ignore the status (unless FAILED)
        _ = ray.get(pending_tasks)

        # XXX(ming): we only keep the latest parameter desc currently
        task_request = TaskRequest(
            task_type=TaskType.ROLLOUT if trainable else TaskType.SIMULATION,
            content=AgentFeedback(
                id=self._id,
                trainable_pairs={
                    aid: (pid, policy.description)
                    for aid, (pid, policy) in policy_dict.items()
                },
                statistics={},
                state_id=task_desc.state_id,
            ),
        )
        self.logger.debug(
            f"send task: {task_request.task_type} {self._id} {task_request.identify}"
        )
        self._coordinator.request.remote(task_request)

        if trainable:
            task_request.task_type = TaskType.OPTIMIZE
            self.logger.debug(
                f"send task: {task_request.task_type} {self._id} {task_request.identify}"
            )
            self._coordinator.request.remote(task_request)

    def get_trainer(self, pid: PolicyID) -> Trainer:
        """Return a registered trainer with given policy id.

        :param PolicyID pid: Policy id.
        :return: A trainer instance.
        """

        return self._trainers[pid]

    def default_policy_id_gen(self, algorithm_conf: Dict[str, Any]) -> str:
        """Generate policy id based on algorithm name and the count of policies. Default to generate policy id as

            `{algorithm_conf[name]}_{len(self._policies)}`.

        :param Dict[str,Any] algorithm_conf: Generate policy id with given algorithm configuration.
        :return: Generated policy id
        """

        return f"{algorithm_conf['name']}_{len(self._policies)}"

    def get_algorithm_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Get algorithm configuration from algorithm candidates. Default to return the first one element of the
        listed value of `algorithm_candidates`.

        :param list args: A list of arg.
        :param dict kwargs: A dict of args.
        :raise: errors.TypeError.
        :return: The algorithm configuration (dict).
        """

        if isinstance(self._algorithm_mapping_func, Callable):
            name = self._algorithm_mapping_func(*args, **kwargs)
            return self.algorithm_candidates[name]
        elif self._algorithm_mapping_func is None:
            return list(self.algorithm_candidates.values())[0]
        else:
            raise errors.TypeError(
                f"Unexpected algorithm mapping function: {self._algorithm_mapping_func}"
            )

    def get_policy_pool_mixture(
        self, weights: Dict[PolicyID, float], agent_id: AgentID, tabular: bool = False
    ) -> Dict[AgentID, Policy]:
        assert list(weights.keys()) == list(self.policies.keys())
        assert np.isclose(1.0, sum(weights.values()))

        class mixed_policy(Policy):
            def __init__(self, observation_space, action_space, policies, weights):
                super(mixed_policy, self).__init__(
                    "mixed", observation_space, action_space
                )
                self._policies = policies
                self._weights = weights

            def compute_actions(
                self, observation: DataTransferType, **kwargs
            ) -> DataTransferType:
                tmp = []
                for pid, weight in self._weights.items():
                    _, batched_action_probs, _ = self._policies[pid].compute_actions(
                        observation, **kwargs
                    )
                    tmp.append(batched_action_probs * weight)
                return _, np.sum(tmp, axis=0), _

            def compute_action(
                self, observation: DataTransferType, **kwargs
            ) -> DataTransferType:
                tmp = []
                for pid, weight in self._weights.items():
                    _, action_probs, _ = self._policies[pid].compute_action(
                        observation, **kwargs
                    )
                    tmp.append(action_probs * weight)
                return _, np.sum(tmp, axis=0), _

        policies = {
            aid: mixed_policy(
                self._observation_spaces[agent_id],
                self._action_spaces[agent_id],
                self.policies,
                weights,
            )
            for aid in self.agent_group
        }
        if tabular:
            policies = {aid: policy.to_tabular() for aid, policy in policies.items()}
        return policies

    def get_policies_with_mapping(
        self, policy_mapping: Dict[AgentID, PolicyID]
    ) -> Dict[AgentID, Policy]:
        return {aid: self._policies[pid] for aid, pid in policy_mapping.items()}

    @abstractmethod
    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Execute policy optimization.

        :param Dict[AgentID,PolicyID] policy_ids: A dict of policies linked to agents registered in `group` required to be optimized
        :param Dict[AgentID,Any] batch: A dict of agent batch, one batch for one policies.
        :param Dict[str,Any] training_config: A dict of training configuration.
        :return: a dict of training feedback
        """
        pass

    @abstractmethod
    def add_policy_for_agent(
        self, env_agent_id: AgentID, trainable
    ) -> Tuple[PolicyID, Policy]:
        """Create new policy and trainer for env agent tagged with `env_agent_id`.

        :param env_agent_id: AgentID, the environment agent id
        :param trainable: bool, tag added policy is trainable or not
        :return: a tuple of policy id and policy
        """

        pass

    @abstractmethod
    def save(self, model_dir: str) -> None:
        """Save policies.

        :param str model_dir: Directory path.
        :return: None
        """

        pass

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Load policies from local storage.

        :param str model_dir: Directory path.
        :return: None
        """

        pass
