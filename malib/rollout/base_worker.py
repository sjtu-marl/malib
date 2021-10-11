"""BaseRolloutWorker integrates a lots of task specified methods required by distributed execution.
Users can implement and register their own rollout worker by inheriting from this class.
"""

from collections import defaultdict
import os
import copy
import time
import traceback

import ray

from malib import settings
from malib.utils.typing import (
    AgentID,
    BufferDescription,
    TaskDescription,
    TaskRequest,
    TaskType,
    RolloutFeedback,
    Status,
    ParameterDescription,
    SimulationDescription,
    RolloutDescription,
    PolicyID,
    Sequence,
    Dict,
    Any,
    Tuple,
    List,
)

from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.agent_interface import AgentInterface
from malib.algorithm.common.policy import Policy
from malib.utils.logger import Logger, get_logger, Log
from malib.utils.stoppers import get_stopper

PARAMETER_GET_TIMEOUT = 3
MAX_PARAMETER_GET_RETRIES = 10


class BaseRolloutWorker:
    def __init__(
        self,
        worker_index: Any,
        env_desc: Dict[str, Any],
        metric_type: str,
        remote: bool = False,
        save: bool = False,
        **kwargs,
    ):
        """Create a rollout worker instance.

        :param Any worker_index: Indicate rollout worker.
        :param Dict[str,Any] env_desc: The environment description.
        :param str metric_type: Name of registered metric handler.
        :param int parallel_num: Number of parallel.
        :param bool remote: Tell this rollout worker work in remote mode or not, default by False.
        :param int save: Whether or not to save the policy models.
        """

        self._worker_index = worker_index
        self._env_description = env_desc
        self._save = save
        self.global_step = 0

        self._coordinator = None
        self._parameter_server = None
        self._offline_dataset = None

        env = env_desc["creator"](**env_desc["config"])
        self._agents = env.possible_agents
        self._kwargs = kwargs

        if remote:
            self.init()

        # interact with environment
        self._agent_interfaces = {
            aid: AgentInterface(
                aid,
                env.observation_spaces[aid],
                env.action_spaces[aid],
                self._parameter_server,
            )
            for aid in self._agents
        }
        self._metric_type = metric_type
        self._remote = remote

        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name=f"rollout_worker_{worker_index}",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **kwargs["exp_cfg"],
        )

    def get_status(self):
        return self._status

    def set_status(self, status):
        if status == self._status:
            return Status.FAILED
        else:
            self._status = status
            return Status.SUCCESS

    @property
    def population(self) -> Dict[PolicyID, Policy]:
        """Return a dict of agent policy pool

        :return: a dict of agent policy id pool
        """

        return {
            agent: list(agent_interface.policies.keys())
            for agent, agent_interface in self._agent_interfaces.items()
        }

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """ Return a remote class for Actor initialization """

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def init(self):
        """Init coordinator in remote mode, parameter server and offline dataset server.

        When worker works in remote mode, this method will be called.
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

    def add_policies(self, task_desc):
        trainable_pairs = task_desc.content.agent_involve_info.trainable_pairs

        population_configs = task_desc.content.agent_involve_info.populations
        parameter_descs = task_desc.content.agent_involve_info.meta_parameter_desc_dict
        for aid, config_seq in population_configs.items():
            # config: (pid, description)
            agent = self._agent_interfaces[aid]
            for pid, pconfig in config_seq:
                if pid not in agent.policies:
                    agent.add_policy(
                        pid,
                        pconfig,
                        parameter_descs[aid].parameter_desc_dict[pid],
                    )
                agent.reset()

            # add policies which need to be trained, and tag it as trainable
        for aid, (pid, description) in trainable_pairs.items():
            agent = self._agent_interfaces[aid]
            try:
                if pid not in agent.policies:
                    agent.add_policy(
                        pid,
                        description,
                        parameter_descs[aid].parameter_desc_dict[pid],
                    )
                agent.reset()
            except Exception as e:
                print(e)
                print(parameter_descs[aid].parameter_desc_dict)

        # check whether there is policy_combinations
        if hasattr(task_desc.content, "policy_combinations"):
            for combination in task_desc.content.policy_combinations:
                for aid, (pid, description) in combination.items():
                    agent = self._agent_interfaces[aid]
                    if pid not in agent.policies:
                        if (
                            parameter_descs[aid].parameter_desc_dict.get(pid, None)
                            is None
                        ):
                            # create a parameter description here
                            parameter_desc = ParameterDescription(
                                time_stamp=time.time(),
                                identify=aid,
                                lock=False,
                                env_id=self._env_description["id"],
                                description=description,
                                id=pid,
                            )
                        else:
                            parameter_desc = parameter_descs[aid].parameter_desc_dict[
                                pid
                            ]
                        agent.add_policy(pid, description, parameter_desc)

    def set_state(self, task_desc: TaskDescription) -> None:
        """Review task description to add new policies and update population distribution.

        :param task_desc: TaskDescription, A task description entity.
        :return: None
        """

        self.add_policies(task_desc)
        if hasattr(task_desc.content, "policy_distribution"):
            self.ready_for_sample(
                policy_distribution=task_desc.content.policy_distribution
            )

    def update_state(self, task_desc: TaskDescription, waiting=False) -> Status:
        """Parse task_desc and check whether it is necessary to update local state.

        :param TaskDescription task_desc: A task description entity.
        :param bool waiting: Update state in sync or async mode
        :return: A status code
        """

        trainable_pairs = task_desc.content.agent_involve_info.trainable_pairs
        populations = task_desc.content.agent_involve_info.populations
        tmp_status = {pid: Status.SUCCESS for pid, _ in trainable_pairs.values()}
        for aid, agent in self._agent_interfaces.items():
            # got policy status then check whether we have need
            if isinstance(task_desc.content, SimulationDescription):
                # update all in population
                for comb in task_desc.content.policy_combinations:
                    pid, _ = comb[aid]
                    parameter_desc = agent.parameter_desc_dict[pid]
                    if not parameter_desc.lock:
                        # fixed policies should wait until locked
                        self.logger.debug(f"pid={pid} for agent={aid} not lock")
                        agent.update_weights([pid], waiting=True)
                        self.logger.debug(f"pid={pid} for agent={aid} locked")
            elif isinstance(task_desc.content, RolloutDescription):
                # update trainable pairs and population
                p_tups = populations[aid]
                # fixed policies should wait until locked
                for pid, _ in p_tups:
                    parameter_desc = agent.parameter_desc_dict[pid]
                    if not parameter_desc.lock:
                        agent.update_weights([pid], waiting=True)
                if aid not in trainable_pairs:
                    continue
                pid, _ = trainable_pairs[aid]
                parameter_desc = agent.parameter_desc_dict[pid]
                if not parameter_desc.lock:
                    status = agent.update_weights([pid], waiting=True)
                    tmp_status[pid] = (
                        Status.LOCKED
                        if Status.LOCKED in status.values()
                        else Status.SUCCESS
                    )
        # return trainable pid
        status = (
            Status.LOCKED if Status.LOCKED in tmp_status.values() else Status.SUCCESS
        )
        self.global_step += 1
        return status

    @Log.method_timer(enable=settings.PROFILING)
    def rollout(self, task_desc: TaskDescription):
        """Collect training data asynchronously and stop it until the evaluation results meet the stopping conditions"""

        stopper = get_stopper(task_desc.content.stopper)(
            config=task_desc.content.stopper_config, tasks=None
        )
        merged_statics = {}
        epoch = 0
        self.set_state(task_desc)
        start_time = time.time()
        total_num_frames = 0
        print_every = 1  # stopper.max_iteration // 3

        # create data table
        trainable_pairs = task_desc.content.agent_involve_info.trainable_pairs
        # XXX(ming): shall we authorize learner to determine the buffer description?
        buffer_desc = BufferDescription(
            env_id=self._env_description["config"]["env_id"],
            agent_id=list(trainable_pairs.keys()),
            policy_id=[pid for pid, _ in trainable_pairs.values()],
            capacity=None,
            data_shapes=self._env_description["config"]["data_shapes"],
            sample_start_size=None,
        )
        ray.get(self._offline_dataset.create_table.remote(buffer_desc))
        while not stopper(merged_statics, global_step=epoch):
            status = self.update_state(task_desc, waiting=False)
            if status == Status.LOCKED:
                break

            trainable_behavior_policies = {
                aid: pid
                for aid, (
                    pid,
                    _,
                ) in task_desc.content.agent_involve_info.trainable_pairs.items()
            }
            # get behavior policies of other fixed agent
            raw_statistics, num_frames = self.sample(
                callback=task_desc.content.callback,
                num_episodes=task_desc.content.num_episodes,
                policy_combinations=[trainable_behavior_policies],
                explore=True,
                fragment_length=task_desc.content.fragment_length,
                role="rollout",
                policy_distribution=task_desc.content.policy_distribution,
                buffer_desc=buffer_desc,
            )

            # merge statis
            res = defaultdict(list)
            for statistics in raw_statistics:
                for k, v in statistics.items():
                    res[k].extend(v)
            res = dict(map(lambda kv: (kv[0], sum(kv[1]) / len(kv[1])), res.items()))
            self.after_rollout(task_desc.content.agent_involve_info.trainable_pairs)
            total_num_frames += num_frames
            time_consump = time.time() - start_time

            # log to tensorboard
            if (epoch + 1) % print_every == 0:
                Logger.info("\tepoch: %s (evaluation) %s", epoch, res)
            for k, v in res.items():
                self.logger.send_scalar(
                    tag=f"evaluation/{k}", content=v, global_step=epoch
                )
            self.logger.send_scalar(
                tag="performance/RFPS/{}".format(os.getpid()),
                content=total_num_frames / time_consump,
                global_step=epoch,
            )
            epoch += 1

        if self._save:
            self.save_model()

        rollout_feedback = RolloutFeedback(
            worker_idx=self._worker_index,
            agent_involve_info=task_desc.content.agent_involve_info,
            statistics=res,
        )
        self.callback(status, rollout_feedback, role="rollout", relieve=True)

    def callback(self, status: Status, content: Any, role: str, relieve: bool):
        if role == "simulation":
            task_req = TaskRequest(
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
                        self._agent_interfaces[agent].policies[pid].state_dict()
                    )
                    _ = ray.get(self._parameter_server.push.remote(parameter_desc))
                self._coordinator.request.remote(
                    TaskRequest(
                        task_type=TaskType.EVALUATE,
                        content=content,
                    )
                )
        if relieve:
            # unlock worker
            self.set_status(Status.IDLE)

    @Log.method_timer(enable=settings.PROFILING)
    def simulation(self, task_desc):
        """Handling simulation task."""

        # set state here
        self.set_state(task_desc)
        combinations = task_desc.content.policy_combinations
        callback = task_desc.content.callback
        agent_involve_info = task_desc.content.agent_involve_info
        # print(f"simulation for {task_desc.content.num_episodes}")
        raw_statistics, num_frames = self.sample(
            callback=callback,
            num_episodes=task_desc.content.num_episodes,
            fragment_length=task_desc.content.max_episode_length,
            policy_combinations=[
                {k: p for k, (p, _) in comb.items()} for comb in combinations
            ],
            explore=False,
            role="simulation",
        )
        Logger.debug("simulation done with return: %s", raw_statistics)
        for statistics, combination in zip(raw_statistics, combinations):
            # merge statistics
            statistics = dict(
                map(lambda kv: (kv[0], sum(kv[1]) / len(kv[1])), statistics.items())
            )
            rollout_feedback = RolloutFeedback(
                worker_idx=self._worker_index,
                agent_involve_info=agent_involve_info,
                statistics=statistics,
                policy_combination=combination,
            )
            task_req = TaskRequest(
                task_type=TaskType.UPDATE_PAYOFFTABLE,
                content=rollout_feedback,
            )
            self._coordinator.request.remote(task_req)
        self.set_status(Status.IDLE)

    def assign_episode_id(self):
        return f"eps-{self._worker_index}-{time.time()}"

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.

        :param policy_distribution: Dict[AgentID, Dict[PolicyID, float]], default by None
        """

        for aid, interface in self._agent_interfaces.items():
            if policy_distribution is None or aid not in policy_distribution:
                pass
            else:
                interface.set_behavior_dist(policy_distribution[aid])
            interface.reset()

    def save_model(self):
        """ Save policy model to log directory. """

        save_dir = os.path.join(
            settings.LOG_DIR,
            self._kwargs["exp_cfg"]["expr_group"],
            self._kwargs["exp_cfg"]["expr_name"],
            "models",
        )
        for aid, interface in self._agent_interfaces.items():
            _save_dir = os.path.join(save_dir, aid)
            interface.save(_save_dir)

    def sample(
        self,
        callback: type,
        num_episodes: int,
        fragment_length: int,
        role: str,
        policy_combinations: List,
        explore: bool = True,
        threaded: bool = True,
        policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None,
        buffer_desc: BufferDescription = None,
    ) -> Tuple[Sequence[Dict[str, List]], int]:
        """Implement your sample logic here, return the collected data and statistics"""

        raise NotImplementedError

    def close(self):
        """ Terminate worker """

        # TODO(ming): store worker's state
        self.logger.info(f"Worker: {self._worker_index} has been terminated.")
        for agent in self._agent_interfaces.values():
            agent.close()

    def after_rollout(self, trainable_pairs):
        """Callback after each iteration"""

        pass
