"""
The coordinator server bridges tasks like training, rollouts, and payoff updates by parsing task requests, generating
new task descriptions, and dispatch them. This coordinator server implementation inherits from `BaseCoordinator`, it is
a special case for large-scale multi-agent learning actually.
"""

import threading

from malib.utils.typing import (
    List,
    TaskDescription,
    TaskRequest,
    TaskType,
    SimulationDescription,
    AgentInvolveInfo,
)
from malib.utils.logger import Logger
from malib.evaluator import get_evaluator, Evaluator
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager
from malib.evaluator.utils.payoff_manager import PayoffManager
from malib.backend.coordinator.base_coordinator import BaseCoordinator


class CoordinatorServer(BaseCoordinator):
    """Coordinator server maintains the payoff matrix and serves for the task assignment."""

    def gen_simulation_task(self, task_request: TaskRequest, matches: List):
        """Generate simulation task for a group of agents"""

        agent_involve_info: AgentInvolveInfo = task_request.content.agent_involve_info

        # load default episode length ?
        max_episode_length = self._configs["evaluation"].get("max_episode_length", 1000)
        num_episodes = self._configs["evaluation"].get("num_episode", 1)
        callback = self._configs["rollout"]["callback"]
        task_desc = TaskDescription(
            task_type=TaskType.SIMULATION,
            content=SimulationDescription(
                callback=callback,
                max_episode_length=max_episode_length,
                agent_involve_info=agent_involve_info,
                policy_combinations=matches,
                num_episodes=num_episodes,  # self._evaluate_config["num_simulation"] * 5
            ),
            state_id=task_request.state_id,
        )
        self.rollout_manager.simulate(task_desc)

    def gen_add_policy_task(self, aid: str, task_request: TaskRequest):
        """Generate policy adding task then dispatch to one agent interface.

        :param str aid: Agent interface id.
        :param Object state_id: A ray object reference
        """

        # tag current task with state_id
        task_desc = TaskDescription(
            task_type=TaskType.ADD_POLICY, content=None, state_id=task_request.state_id
        )
        self._training_manager.add_policy(aid, task_desc)

    def __init__(
        self,
        **kwargs,
    ):
        """Create a coordinator server instance."""

        BaseCoordinator.__init__(self)

        self._configs = kwargs
        self._terminate = False
        self._pending_trainable_pairs = {}
        self._exp_cfg = kwargs["exp_cfg"]

        # maintain the population sets.
        self._populations = {
            agent: set()
            for agent in self._configs["env_description"]["possible_agents"]
        }
        assert (
            len(self._populations) > 0
        ), "no possible agents detected, please specify it in the env_description"

        # payoff manager responses for the payoff management of all agents
        self.payoff_manager = PayoffManager(
            self._configs["env_description"]["possible_agents"],
            kwargs["exp_cfg"],
            solve_method="fictitious_play"
            if len(self._populations) < 3
            else "alpharank",
        )

        # hyper_evaluator: determine global convergence achievement or not
        self._hyper_evaluator: Evaluator = get_evaluator(
            self._configs["global_evaluator"]["name"]
        )(**self._configs["global_evaluator"]["config"])

        self.training_manager = None
        self.rollout_manager = None
        self.task_mode = kwargs["task_mode"]
        self.request_lock = threading.Lock()

    @property
    def hyper_evaluator(self) -> Evaluator:
        return self._hyper_evaluator

    @property
    def payoff_manager(self) -> PayoffManager:
        return self._payoff_manager

    @payoff_manager.setter
    def payoff_manager(self, value):
        self._payoff_manager = value

    def init_managers(self):
        self.training_manager = TrainingManager(
            algorithms=self._configs["algorithms"],
            env_desc=self._configs["env_description"],
            interface_config=self._configs["training"]["interface"],
            training_agent_mapping=self._configs["agent_mapping_func"],
            training_config=self._configs["training"]["config"],
            exp_cfg=self._exp_cfg,
        )

        self.rollout_manager = RolloutWorkerManager(
            num_worker=self.training_manager.get_agent_interface_num(),
            agent_mapping_func=self._configs["agent_mapping_func"],
            rollout_config=self._configs["rollout"],
            env_desc=self._configs["env_description"],
            exp_cfg=self._exp_cfg,
        )

        Logger.info(
            "set worker num as {}".format(
                self.training_manager.get_agent_interface_num()
            )
        )

    def start(self):
        """Start coordinator server, init TrainingManager and RolloutManager."""
        self.init_managers()
        self.training_manager.init(state_id=self.generate_task_id())

        Logger.info("Coordinator server started")

    def request(self, task_request: TaskRequest):
        """Handling task request"""

        # call request by name
        with self.request_lock:
            # Logger.debug("request: {}".format(task_request.task_type))
            generic_task_handler = getattr(
                CoordinatorServer,
                "_request_{}".format(task_request.task_type.value),
                None,
            )

            if generic_task_handler:
                generic_task_handler(self, task_request)
            else:
                raise AttributeError(
                    f"Missing handler for task type {task_request.task_type.value}"
                )

    def is_terminate(self):
        return self._terminate

    def terminate(self):
        self._terminate = True  # hard terminate
        self.training_manager.terminate()
        self.rollout_manager.terminate()
