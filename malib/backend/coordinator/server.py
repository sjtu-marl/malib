"""
The coordinator server bridges tasks like training, rollouts, and payoff updates by parsing task requests, generating
new task descriptions, and dispatch them. This coordinator server implementation inherits from `BaseCoordinator`, it is
a special case for large-scale multi-agent learning actually.
"""

import threading

import copy
from typing import List, Dict

import ray

from malib import settings
from malib.utils.formatter import pretty_print as pp
from malib.utils.typing import (
    AgentID,
    TaskDescription,
    TaskRequest,
    TaskType,
    RolloutDescription,
    TrainingDescription,
    EvaluateResult,
    TrainingFeedback,
    SimulationDescription,
    AgentInvolveInfo,
    BColors,
)
from malib.utils.logger import get_logger
from malib.evaluator import get_evaluator, Evaluator
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager
from malib.evaluator.utils.payoff_manager import PayoffManager
from malib.backend.coordinator.base_coordinator import BaseCoordinator


@ray.remote
class CoordinatorServer(BaseCoordinator):
    """Coordinator server maintains the payoff matrix and serves for the task assignment."""

    def push(self, **kwargs):
        pass

    def pull(self, **kwargs):
        pass

    def __init__(
        self,
        **kwargs,
    ):
        """Create a coordinator server instance."""

        BaseCoordinator.__init__(self)

        self._configs = kwargs
        self._terminate = False
        self._pending_trainable_pairs = {}

        self._offline: bool = self._configs["training"]["config"]["offline"]

        # maintain the population sets.
        self._populations = {
            agent: set()
            for agent in self._configs["env_description"]["possible_agents"]
        }
        assert (
            len(self._populations) > 0
        ), "no possible agents detected, please specify it in the env_description"
        # payoff manager responses for the payoff management of all agents
        self._payoff_manager = PayoffManager(
            self._configs["env_description"]["possible_agents"], kwargs["exp_cfg"]
        )
        # hyper_evaluator: determine global convergence achievement or not
        self._hyper_evaluator: Evaluator = get_evaluator(
            self._configs["global_evaluator"]["name"]
        )(**self._configs["global_evaluator"]["config"])

        self._rollout_worker_manager = None
        self._training_manager = None
        self._lock = threading.Lock()
        self._exp_cfg = kwargs["exp_cfg"]
        self._logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            name="coordinator",
            **kwargs["exp_cfg"],
        )

    def start(self):
        self._training_manager = TrainingManager(
            algorithms=self._configs["algorithms"],
            rewards=self._configs["rewards"],
            env_desc=self._configs["env_description"],
            interface_config=self._configs["training"]["interface"],
            training_agent_mapping=self._configs["agent_mapping_func"],
            training_config=self._configs["training"]["config"],
            exp_cfg=self._exp_cfg,
        )
        if not self._offline:
            # one training interface one rollout worker
            self._configs["rollout"][
                "worker_num"
            ] = self._training_manager.get_agent_interface_num()
            self._rollout_worker_manager = RolloutWorkerManager(
                rollout_config=self._configs["rollout"],
                env_desc=self._configs["env_description"],
                exp_cfg=self._exp_cfg,
            )
        self._training_manager.init()

        self._logger.info("Coordinator server started")

    def pre_launching(self, init_config):
        # if init_config["load_model"]:
        #     self.request(
        #         TaskRequest(
        #             task_type=TaskType.LOAD_MODEL,
        #             content=init_config["model_path"],
        #         )
        #     )
        #     self.request(
        #         Tasks
        #     )
        pass

    @staticmethod
    def task_handler_register(cls):
        from functools import wraps

        print("Registering")

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(*args, **kwargs)

            setattr(cls, func.__name__, func)
            return func

        return decorator

    def request(self, task_request: TaskRequest):
        """ Handling task request """

        if task_request.task_type == TaskType.SIMULATION:
            if self._offline:
                return
            # content is TrainingFeedback
            task_request = self._training_manager.retrieve_information(task_request)
            pending_matches = []
            for (
                env_aid,
                ptup,
            ) in task_request.content.agent_involve_info.trainable_pairs.items():
                self._pending_trainable_pairs[env_aid] = ptup
                # create matches here
                pending_matches.extend(
                    self._payoff_manager.get_pending_matchups(env_aid, *ptup)
                )
            if len(pending_matches) > 0:
                self.gen_simulation_task(task_request, pending_matches)
        elif task_request.task_type == TaskType.EVALUATE:
            """Requests from rollout worker after rollout tasks done, or agent.AgentInterface after optimize tasks done.
            Evaluate task here aims to determine whether to do simulation or terminate task directly.
            """
            populations = task_request.content.agent_involve_info.populations
            trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
            pending_matches = []
            for env_aid, ptup in trainable_pairs.items():
                pending_matches.extend(
                    self._payoff_manager.get_pending_matchups(env_aid, *ptup)
                )

            if len(pending_matches) == 0:
                self._logger.warning(
                    BColors.WARNING + "repeated policy id detected!" + BColors.ENDC
                )
                for env_aid, ptup in trainable_pairs.items():
                    self._pending_trainable_pairs[env_aid] = ptup
            else:
                self.gen_simulation_task(task_request, pending_matches)
        elif task_request.task_type == TaskType.UPDATE_PAYOFFTABLE:
            """ Update payoff table after simulations, and then generate new policies """
            with self._lock:
                self.update_payoff_table(task_request)
        elif task_request.task_type == TaskType.ROLLOUT:
            # NOTE(zbzhu): if offline training, do not rollout
            if self._offline:
                return
            task_request = self._training_manager.retrieve_information(task_request)
            self.gen_rollout_task(task_request)
        elif task_request.task_type == TaskType.OPTIMIZE:
            task_request = self._training_manager.retrieve_information(task_request)
            self.gen_optimization_task(task_request.content.agent_involve_info)
        elif task_request.task_type == TaskType.TERMINATE:
            self._terminate = True
        elif task_request.task_type in TaskType:
            generic_task_handler = getattr(self, task_request.task_type, None)
            if generic_task_handler:
                generic_task_handler(task_request)
            else:
                raise AttributeError(
                    f"Missing handler for task type {task_request.task_type}"
                )
        else:
            raise TypeError(f"Unexpected task type: {task_request.task_type}")

    def update_payoff_table(self, task_request: TaskRequest):
        """Update payoff table, add evaluated policies and generate new policies
        if all policies finished their simulation.
        """

        rollout_feedback = task_request.content
        self._payoff_manager.update_payoff(rollout_feedback)

        agent_involve_info = rollout_feedback.agent_involve_info
        population_mapping = agent_involve_info.populations
        # filter policy configuration
        population_mapping = {
            mpid: [pt[0] for pt in ptlist]
            for mpid, ptlist in population_mapping.items()
        }

        for env_aid, p_tuple in agent_involve_info.trainable_pairs.items():
            self._pending_trainable_pairs[env_aid] = p_tuple
            if p_tuple[0] not in population_mapping[env_aid]:
                population_mapping[env_aid].append(p_tuple[0])
        for env_aid, p_tuple in self._pending_trainable_pairs.items():
            if p_tuple[0] not in population_mapping[env_aid]:
                population_mapping[env_aid].append(p_tuple[0])

        all_done = self._payoff_manager.check_done(population_mapping)
        if all_done and len(self._pending_trainable_pairs) == len(self._populations):
            self._logger.info("All pending payoffs have been updated")

            self._logger.debug(
                f"sending policy adding task with pending trainable pairs:"
                f"\n{pp(self._pending_trainable_pairs)}"
            )
            # gen new population mapping
            new_population_mapping = copy.copy(population_mapping)
            # check not trainable
            for agent, ele in new_population_mapping.items():
                if (
                    self._pending_trainable_pairs[agent][0]
                    in new_population_mapping[agent]
                ):
                    continue
                new_population_mapping[agent].append(
                    self._pending_trainable_pairs[agent][0]
                )
            state_id = ray.put(new_population_mapping)
            # TODO(ming): require a better implementation, or move this branch to payoff_manager
            if len(new_population_mapping) > 1:
                equilibrium = self._payoff_manager.compute_equilibrium(
                    new_population_mapping
                )
                self._payoff_manager.update_equilibrium(
                    new_population_mapping, equilibrium
                )
                # oracle payoffs: payoff aggregation with a equilibrium (Nash / Coordination / ...)
                oracle_payoffs: Dict[AgentID, float] = self._payoff_manager.aggregate(
                    equilibrium=equilibrium
                )
                # weighted payoffs: payoff aggregation with the learned best response and fixed opponent policies
                weighted_payoffs: Dict[AgentID, float] = self._payoff_manager.aggregate(
                    equilibrium=equilibrium,
                    brs={
                        aid: pid
                        for aid, (pid, _) in self._pending_trainable_pairs.items()
                    },
                )
                # XXX(ming): PSRO is a special case, require improvement
                if self._configs["global_evaluator"]["name"] == "psro":
                    exp = self._training_manager.get_exp(equilibrium)
                    print("######### payoff:")
                    print(list(self._payoff_manager.payoffs.values())[0].table)
                    print("######### equilibriumn:", equilibrium)
                    print("######### exploitability:", exp)
                    self._logger.send_scalar(
                        tag="metric/exp",
                        content=exp,
                        global_step=len(equilibrium["player_0"]),
                    )
            else:
                weighted_payoffs = None
                oracle_payoffs = None
            evaluate_result = self._hyper_evaluator.evaluate(
                # content here should be
                task_request.content,
                weighted_payoffs=weighted_payoffs,
                oracle_payoffs=oracle_payoffs,
                trainable_mapping=self._pending_trainable_pairs,
            )
            if evaluate_result[EvaluateResult.CONVERGED]:
                self._terminate = True
            else:

                self._pending_trainable_pairs = {}
                for aid in self._training_manager.groups:
                    self.gen_add_policy_task(aid, state_id)
        else:
            self._logger.warning(
                f"payoff evaluation for policies doesn't finish yet, skip policy adding."
            )

    def gen_simulation_task(self, task_request: TaskRequest, matches: List):
        """ Generate simulation task for a group of agents """

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
            state_id=None,
        )
        self._rollout_worker_manager.simulate(task_desc)

    def gen_optimization_task(self, agent_involve_info: AgentInvolveInfo):
        task_desc = TaskDescription(
            task_type=TaskType.OPTIMIZE,
            content=TrainingDescription(
                agent_involve_info=agent_involve_info,
                stopper=self._configs["training"]["config"]["stopper"],
                stopper_config=self._configs["training"]["config"]["stopper_config"],
                batch_size=self._configs["training"]["config"]["batch_size"],
                update_interval=self._configs["training"]["config"]["update_interval"],
            ),
            state_id=None,
        )
        self._training_manager.optimize(task_desc)

    def gen_add_policy_task(self, aid: str, state_id):
        """Generate policy adding task then dispatch to one agent interface.

        :param str aid: Agent interface id.
        :param Object state_id: A ray object reference
        """

        # tag current task with state_id
        task_desc = TaskDescription(
            task_type=TaskType.ADD_POLICY, content=None, state_id=state_id
        )
        self._training_manager.add_policy(aid, task_desc)

    def gen_rollout_task(self, task_request: TaskRequest):
        """ Generate rollout task by parsing task request from AgentActor """

        assert isinstance(task_request.content, TrainingFeedback)

        populations = task_request.content.agent_involve_info.populations
        population_mapping = {}
        for k, v in populations.items():
            assert len(v) > 0, v
            population_mapping[k] = [p[0] for p in v]
        agent_involve_info = task_request.content.agent_involve_info

        if all([len(p_list) for p_list in population_mapping.values()]):
            policy_distribution = self._payoff_manager.get_equilibrium(
                population_mapping
            )
            for env_aid, (pid, _) in agent_involve_info.trainable_pairs.items():
                policy_distribution[env_aid] = {pid: 1.0}
            # since in meta_policy this is a default_dict with value 0.0
        else:
            policy_distribution = {}

        rollout_config = self._configs["rollout"]
        task = TaskDescription(
            task_type=TaskType.ROLLOUT,
            content=RolloutDescription(
                agent_involve_info=agent_involve_info,
                policy_distribution=policy_distribution,
                fragment_length=rollout_config["fragment_length"],
                num_episodes=rollout_config["num_episodes"],
                stopper=rollout_config["stopper"],
                stopper_config=rollout_config["stopper_config"],
                terminate_mode=rollout_config["terminate"],
                mode=rollout_config["mode"],
                callback=rollout_config["callback"],
                episode_seg=rollout_config["episode_seg"],
            ),
            state_id=None,
        )

        self._rollout_worker_manager.rollout(task_desc=task)

        if rollout_config.get("test_num_episodes", 0) > 0:
            task = TaskDescription(
                task_type=TaskType.ROLLOUT,
                content=RolloutDescription(
                    agent_involve_info=agent_involve_info,
                    policy_distribution=policy_distribution,
                    fragment_length=rollout_config["fragment_length"],
                    num_episodes=rollout_config["test_num_episodes"],
                    stopper="none",
                    stopper_config=rollout_config["stopper_config"],
                    terminate_mode=rollout_config["terminate"],
                    mode=rollout_config["mode"],
                    callback=rollout_config["callback"],
                    episode_seg=rollout_config["episode_seg"],
                ),
                state_id=None,
            )

            self._rollout_worker_manager.rollout(task_desc=task, test=True)

    def is_terminate(self):
        return self._terminate

    def terminate(self):
        self._training_manager.terminate()
        self._rollout_worker_manager.terminate()
