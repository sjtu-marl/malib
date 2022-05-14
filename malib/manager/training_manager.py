"""
Implementation of training manager, which is responsible for a group of training agent interfaces.
"""

from collections import defaultdict
import os
from typing import Dict, Any, Callable, Sequence

import psutil
import ray

from malib import settings
from malib.agent import get_training_agent
from malib.agent.agent_interface import AgentFeedback, AgentTaggedFeedback
from malib.gt.algos.exploitability import measure_exploitability
from malib.utils.logger import Log, Logger
from malib.utils.typing import (
    List,
    Tuple,
    PolicyConfig,
    TaskType,
    AgentID,
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
        env_desc: Dict[str, Any],
        interface_config: Dict[str, Any],
        training_agent_mapping: Callable,
        training_config: Dict[str, Any],
        exp_cfg: Dict[str, Any],
    ):
        """Create an TrainingManager instance which is responsible for the multi agent training
        tasks execution and rollout task requests sending.

        :param Dict[str,Any] algorithms: The algorithms configuration candidates.
        :param Dict[str,Any] env_desc: The description for environment generation.
        :param Callable training_agent_mapping: The mapping function maps agent id to training interface id.
        :param Dict[AgentID,Dict[str,Any]] training_config: The agent configuration dictionary.
        :param Dict[str,Any] exp_cfg: Experiment description.
        """

        self._env_description = env_desc
        self._training_config = training_config
        self._training_agent_mapping = training_agent_mapping

        # interface config give the agent type used here and the group mapping if needed
        agent_type = interface_config["type"]
        agent_cls = get_training_agent(agent_type)

        # training agent mapping function is enough for
        training_agent_mapping = training_agent_mapping or (lambda agent_id: agent_id)

        self._agents = {}
        agent_groups = defaultdict(lambda: [])
        sorted_env_agents = sorted(env_desc["possible_agents"])
        for agent in sorted_env_agents:
            rid = training_agent_mapping(agent)
            agent_groups[rid].append(agent)

        # FIXME(ming): resource configuration is not available now, will open in the next version
        if training_config.get("use_cuda", False):
            num_gpus = 1 / len(agent_groups)
        else:
            num_gpus = 0.0
        agent_cls = agent_cls.as_remote(num_gpus=num_gpus)

        for rid, agents in agent_groups.items():
            self._agents[rid] = agent_cls.options(max_concurrency=100).remote(
                assign_id=rid,
                env_desc=env_desc,
                algorithm_candidates=algorithms,
                training_agent_mapping=training_agent_mapping,
                observation_spaces=interface_config["observation_spaces"],
                action_spaces=interface_config["action_spaces"],
                exp_cfg=exp_cfg,
                use_init_policy_pool=interface_config["use_init_policy_pool"],
                population_size=interface_config["population_size"],
                algorithm_mapping=interface_config["algorithm_mapping"],
                governed_agents=agents,
            )

        # for env_aid in sorted_env_agents:
        #     training_aid = training_agent_mapping(env_aid)
        #     if isinstance(training_aid, str):
        #         training_aids = [training_aid]
        #     else:
        #         training_aids = training_aid

        #     for training_aid in training_aids:
        #         if training_aid not in self._agents:
        #             self._agents[training_aid] = agent_cls.options(
        #                 max_concurrency=100
        #             ).remote(
        #                 assign_id=training_aid,
        #                 env_desc=env_desc,
        #                 algorithm_candidates=algorithms,
        #                 training_agent_mapping=training_agent_mapping,
        #                 observation_spaces=interface_config["observation_spaces"],
        #                 action_spaces=interface_config["action_spaces"],
        #                 exp_cfg=exp_cfg,
        #                 use_init_policy_pool=interface_config["use_init_policy_pool"],
        #                 population_size=interface_config["population_size"],
        #                 algorithm_mapping=interface_config["algorithm_mapping"],
        #             )
        #         # register trainable env agents
        #         self._agents[training_aid].register_env_agent.remote(env_aid)
        #         if agent_groups.get(training_aid) is None:
        #             agent_groups[training_aid] = []
        #         agent_groups[training_aid].append(env_aid)

        _ = ray.get([agent.start.remote() for agent in self._agents.values()])

        # training_agent -> env_agents
        self._groups = agent_groups

        Logger.info(
            f"training manager launched, {len(self._agents)} learner(s) created"
        )

    def get_agent_interface_num(self) -> int:
        """Get the number of agent interfaces."""

        return len(self._agents)

    def init(self, state_id: str) -> None:
        """Initialize all training agents. Add fixed policies for them.

        :return: None
        """
        tasks = []
        # add policy
        for rid, agent_interface in self._agents.items():
            tasks.append(
                agent_interface.add_policy.remote(
                    TaskDescription(
                        task_type=TaskType.ADD_POLICY,
                        content=TrainingDescription(
                            agent_involve_info=AgentInvolveInfo(
                                training_handler=rid,
                                trainable_pairs=dict.fromkeys(self.groups[rid], None),
                                populations={},
                            ),
                        ),
                        state_id=state_id,
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

    # @Log.method_timer(enable=settings.PROFILING)
    def optimize(self, task: TaskDescription) -> None:
        """Dispatch optimization tasks to training agent interface.

        :param TaskDescription task: A task description entity.
        :return: None
        """

        agent_involve_info = task.content.agent_involve_info
        interface = self._agents[agent_involve_info.training_handler]
        interface.train.remote(task, self._training_config)

    def _get_population_desc(
        self, state_id: str
    ) -> Dict[AgentID, List[Tuple[PolicyID, PolicyConfig]]]:
        """Return stationary population description with given state_id which is related to a muted object stored as a
        Ray object.

        :return: A dictionary describes the population
        """

        tasks = [agent.get_stationary_state.remote() for agent in self._agents.values()]
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

    # @Log.method_timer(enable=settings.PROFILING)
    def retrieve_information(self, task_request: TaskRequest) -> TaskRequest:
        """Fill task request with a training feedback if possible. If there already is a training feedback entity
        assigned to `task_request.content`, return the original task request directly.

        :param TaskRequest task_request: A task request entity.
        :raise: TypeError
        :return: A task request entity filled with training feedback.
        """

        if isinstance(task_request.content, AgentFeedback):
            populations = self._get_population_desc(task_request.state_id)
            tasks = [
                agent.require_parameter_desc.remote() for agent in self._agents.values()
            ]

            # state_id = task_request.content.get("state_id", None) or ray.put(populations)
            meta_parameter_desc_dict = {}
            for desc in ray.get(tasks):
                meta_parameter_desc_dict.update(desc)
                # meta_parameter_desc_dict[desc.meta_pid] = desc

            task_request.content = TrainingFeedback(
                agent_involve_info=AgentInvolveInfo(
                    training_handler=task_request.content.id,
                    env_id=self._env_description["config"]["env_id"],
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
            ray.kill(agent)
        del self._agents

    def get_exp(self, policy_distribution):
        """Compute exploitability"""

        # XXX(ming): may we migrate this function to (PSRO) evaluator
        populations = {}
        for aid, agent in self._agents.items():
            populations[aid] = ray.get(agent.get_policies.remote())
        nashconv, _ = measure_exploitability(
            self._env_description["config"]["env_id"],
            populations=populations,
            policy_mixture_dict=policy_distribution,
        )

        return nashconv
