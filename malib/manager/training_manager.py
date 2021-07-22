"""
Implementation of training manager, which is responsible for a group of training agent interfaces.
"""

import os
from typing import Dict, Any, Callable, Sequence

import psutil
import ray

from malib import settings
from malib.agent import get_training_agent
from malib.agent.agent_interface import AgentFeedback, AgentTaggedFeedback
from malib.gt.algos.exploitability import measure_exploitability
from malib.utils.logger import get_logger, Log
from malib.utils.typing import (
    List,
    TaskType,
    PolicyID,
    TaskRequest,
    TrainingFeedback,
    TaskDescription,
    AgentInvolveInfo,
    TrainingDescription,
)


class TrainingManager:
    def __init__(
        self,
        algorithms: Dict[str, Any],
        rewards: Dict[str, Any],
        env_desc: Dict[str, Any],
        interface_config: Dict[str, Any],
        training_agent_mapping: Callable,
        training_config: Dict[str, Any],
        exp_cfg: Dict[str, Any],
    ):
        """Create an TrainingManager instance which is responsible for the multi agent training
        tasks execution and rollout task requests sending.

        :param Dict[str,Any] algorithms: The algorithms configuration candidates.
        :param Dict[str, Any] rewards: The rewards configuration candidates, use environment reward by default.
        :param Dict[str,Any] env_desc: The description for environment generation.
        :param Callable training_agent_mapping: The mapping function maps agent id to training interface id.
        :param Dict[AgentID,Dict[str,Any]] training_config: The agent configuration dictionary.
        :param Dict[str,Any] exp_cfg: Experiment description.
        """

        self._env_description = env_desc
        self._training_config = training_config
        self._training_agent_mapping = training_agent_mapping
        self._offline = training_config["offline"]

        # interface config give the agent type used here and the group mapping if needed
        agent_type = interface_config["type"]
        agent_cls = get_training_agent(agent_type)

        # training agent mapping function is enough for
        training_agent_mapping = training_agent_mapping or (lambda agent_id: agent_id)

        # FIXME(ming): resource configuration is not available now, will open in the next version
        agent_cls = agent_cls.as_remote(
            **interface_config["worker_config"]
            # num_cpus=None,
            # num_gpus=None,
            # memory=None,
            # object_store_memory=None,
            # resources=None,
        )

        self._agents = {}
        groups = (
            {}
        )  # mapping from training agent id to environment agent id (many to many)
        sorted_env_agents = sorted(env_desc["possible_agents"])
        for env_aid in sorted_env_agents:
            training_aid = training_agent_mapping(env_aid)
            if isinstance(training_aid, str):
                training_aids = [training_aid]
            else:
                training_aids = training_aid

            for training_aid in training_aids:
                if training_aid not in self._agents:
                    self._agents[training_aid] = agent_cls.options(
                        max_concurrency=100
                    ).remote(
                        training_aid,
                        env_desc,
                        algorithms,
                        rewards,
                        training_agent_mapping,
                        interface_config["observation_spaces"],
                        interface_config["action_spaces"],
                        exp_cfg,
                        interface_config["population_size"],
                        interface_config["algorithm_mapping"],
                    )
                # register trainable env agents
                self._agents[training_aid].register_env_agent.remote(env_aid)
                if groups.get(training_aid) is None:
                    groups[training_aid] = []
                groups[training_aid].append(env_aid)

        _ = ray.get([agent.start.remote() for agent in self._agents.values()])

        self._groups = groups
        self.proc = psutil.Process(os.getpid())

        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name="training_manager",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )
        self.logger.debug(f"{len(self._agents)} agents have been created")

    def get_agent_interface_num(self) -> int:
        return len(self._agents)

    @Log.method_timer(enable=settings.PROFILING)
    def init(self) -> None:
        """Initialize all training agents. Add fixed policies for them.

        :return: None
        """
        tasks = []
        # add policy
        for aid, agent_interface in self._agents.items():
            tasks.append(
                agent_interface.add_policy.remote(
                    TaskDescription(
                        task_type=TaskType.ADD_POLICY,
                        content=TrainingDescription(
                            agent_involve_info=AgentInvolveInfo(
                                training_handler=aid,
                                trainable_pairs=dict.fromkeys(self.groups[aid], None),
                                populations={},
                            ),
                        ),
                        state_id=None,
                    )
                )
            )
        _ = ray.get(tasks)

    @property
    def groups(self) -> Dict[str, List]:
        """Return agent groups.

        :return: A dict mapping training agent id to a group of environment agent ids.
        """

        return self._groups

    def add_policy(self, interface_id: str, task: TaskDescription) -> None:
        """Dispatch policy adding task to training agent interface tagged with `interface_id` with give task description.

        :param str interface_id: Training agent interface id.
        :param TaskDescription task: A task description whose content is a `TrainingDescription`.
        :return: None
        """

        agent_interface = self._agents[interface_id]
        agent_interface.add_policy.remote(task)

    @Log.method_timer(enable=settings.PROFILING)
    def optimize(self, task: TaskDescription) -> None:
        """Dispatch optimization tasks to training agent interface.

        :param TaskDescription task: A task description entity.
        :return: None
        """

        agent_involve_info = task.content.agent_involve_info
        interface = self._agents[agent_involve_info.training_handler]
        interface.train.remote(task, self._training_config)

    def _get_population_desc(
        self, state_id: ray.ObjectID
    ) -> Dict[PolicyID, Sequence[Any]]:
        """Return stationary population description with given state_id which is related to a muted object stored as a
        Ray object.

        :param ray.ObjectID state_id: A muted object id.
        :return: A dictionary describes the population
        """

        tasks = [
            agent.get_stationary_state.remote(state_id)
            for agent in self._agents.values()
        ]
        populations = {}

        while len(tasks) > 0:
            dones, tasks = ray.wait(tasks)
            for done in dones:
                agent_tagged_feedback = ray.get(done)
                assert isinstance(
                    agent_tagged_feedback, AgentTaggedFeedback
                ), agent_tagged_feedback
                populations.update(agent_tagged_feedback.content)

        return populations

    @Log.method_timer(enable=settings.PROFILING)
    def retrieve_information(self, task_request: TaskRequest) -> TaskRequest:
        """Fill task request with a training feedback if possible. If there already is a training feedback entity
        assigned to `task_request.content`, return the original task request directly.

        :param TaskRequest task_request: A task request entity.
        :raise: TypeError
        :return: A task request entity filled with training feedback.
        """

        if isinstance(task_request.content, AgentFeedback):
            populations = self._get_population_desc(task_request.content.state_id)
            tasks = [
                agent.require_parameter_desc.remote(task_request.content.state_id)
                for agent in self._agents.values()
            ]

            # state_id = task_request.content.get("state_id", None) or ray.put(populations)
            meta_parameter_desc_dict = {}
            for desc in ray.get(tasks):
                meta_parameter_desc_dict.update(desc)
                # meta_parameter_desc_dict[desc.meta_pid] = desc

            task_request.content = TrainingFeedback(
                agent_involve_info=AgentInvolveInfo(
                    training_handler=task_request.content.id,
                    env_id=self._env_description["id"],
                    populations=populations,
                    trainable_pairs=task_request.content.trainable_pairs,
                    meta_parameter_desc_dict=meta_parameter_desc_dict,
                ),
                statistics=task_request.content.statistics,
            )
        elif isinstance(task_request.content, TrainingFeedback):
            pass
        else:
            raise TypeError(f"Unexpected task content: {task_request.content}")

        return task_request

    def terminate(self) -> None:
        """Terminate all training agent actor.

        :return: None
        """

        for agent in self._agents.values():
            # TODO(ming): save interval state
            agent.exit_actor()
        del self._agents

    def get_exp(self, policy_distribution):
        """Compute exploitability"""

        # XXX(ming): may we migrate this function to (PSRO) evaluator
        populations = {}
        for aid, agent in self._agents.items():
            populations[aid] = ray.get(agent.get_policies.remote())
        nashconv, _ = measure_exploitability(
            "leduc_poker",
            populations=populations,
            policy_mixture_dict=policy_distribution,
        )

        return nashconv
