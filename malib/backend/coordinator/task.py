import threading
import copy

from dataclasses import dataclass, field
from collections import defaultdict

from malib.utils.logging import Logger
from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    TaskType,
    TrainingFeedback,
    TrainingDescription,
    RolloutDescription,
    List,
    EvaluateResult,
    Dict,
)
from malib.utils.tasks_register import task_handler_register, helper_register
from malib.utils.general import frozen_data
from malib.backend.coordinator.server import CoordinatorServer


@dataclass
class TaskCache:
    matches: List = field(default_factory=list)
    trainable_pairs: Dict = field(default_factory=dict)
    population_mapping: Dict = field(default_factory=dict)
    queue_size: int = 1
    mode: str = "bulk_sync"

    def __post_init__(self):
        self.plock = threading.Lock()
        self.mlock = threading.Lock()
        self.tlock = threading.Lock()
        self.n_pending_training_task = 0

        # self.pending_training = defaultdict(lambda: 0)

    def clean(self):
        with self.plock:
            self.population_mapping = {}
        with self.mlock:
            self.matches = []
        with self.tlock:
            self.trainable_pairs = {}

    def get_population_mapping(self):
        with self.plock:
            return self.population_mapping

    def get_trainable_pairs(self):
        with self.tlock:
            return self.trainable_pairs

    def get_matches(self):
        with self.mlock:
            return self.matches

    def reset_matches(self):
        with self.mlock:
            self.matches = []

    def update_population_mapping(self, v):
        with self.plock:
            for k, _v in v.items():
                if k not in self.population_mapping:
                    self.population_mapping[k] = []
                for e in _v:
                    if e not in self.population_mapping[k]:
                        self.population_mapping[k].append(e)

    def extend_matches(self, v):
        with self.mlock:
            self.matches.extend(v)

    def update_trainable_pairs(self, v):
        with self.tlock:
            self.trainable_pairs.update(v)

    def reset_trainable_pairs(self):
        with self.tlock:
            self.trainable_pairs = {}
            self.n_pending_training_task = 0

    def all_training_done(self):
        return self.n_pending_training_task == 0


@task_handler_register(cls=CoordinatorServer, link=TaskType.OPTIMIZE.value)
def _request_optimize(coordinator: CoordinatorServer, task_request: TaskRequest):
    task_request = coordinator.training_manager.retrieve_information(task_request)
    task_desc = TaskDescription(
        task_type=TaskType.OPTIMIZE,
        content=TrainingDescription(
            agent_involve_info=task_request.content.agent_involve_info,
            stopper=coordinator._configs["training"]["config"]["stopper"],
            stopper_config=coordinator._configs["training"]["config"].get(
                "stopper_config", None
            ),
            batch_size=coordinator._configs["training"]["config"]["batch_size"],
            update_interval=coordinator._configs["training"]["config"][
                "update_interval"
            ],
        ),
        state_id=None,
    )
    coordinator.training_manager.optimize(task_desc)


@task_handler_register(cls=CoordinatorServer, link=TaskType.SIMULATION.value)
def _request_simulation(coordinator: CoordinatorServer, task_request: TaskRequest):

    Logger.debug("request for simulation")
    # fill message for this request
    task_request = coordinator.training_manager.retrieve_information(task_request)

    # generate pending matches
    pending_matches = []
    pending_trainable_pairs = {}
    pending_population = {}

    # cache task related information
    for (
        env_aid,
        p_tup,
    ) in task_request.content.agent_involve_info.trainable_pairs.items():
        pending_trainable_pairs[env_aid] = p_tup[0]
        pending_population[env_aid] = [p_tup[0]]
        pending_matches.extend(
            coordinator.payoff_manager.get_pending_matchups(env_aid, *p_tup)
        )
    if len(pending_matches) > 0:
        coordinator.gen_simulation_task(task_request, pending_matches)

    if coordinator.task_cache.get(task_request.state_id) is None:
        coordinator.task_cache[task_request.state_id] = TaskCache(
            mode=task_request.computing_mode
        )
        Logger.debug(
            "generate task cache for state_id={}".format(task_request.state_id)
        )
    task_cache = coordinator.task_cache[task_request.state_id]
    task_cache.extend_matches(pending_matches)
    task_cache.update_trainable_pairs(pending_trainable_pairs)

    # update population mapping with pending trainable_pairs
    task_cache.update_population_mapping(pending_population)


@task_handler_register(cls=CoordinatorServer, link=TaskType.EVALUATE.value)
def _request_evaluation(coordinator: CoordinatorServer, task_request: TaskRequest):
    Logger.debug("rollout done, request for evaluation")
    trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
    pending_matches = []
    pending_trainable_pairs = {}
    pending_population = {}

    for env_aid, ptup in trainable_pairs.items():
        pending_matches.extend(
            coordinator.payoff_manager.get_pending_matchups(env_aid, *ptup)
        )
        pending_population[env_aid] = [ptup[0]]

    if len(pending_matches) == 0:
        Logger.warning("repeated policy id detected!")
        for env_aid, ptup in trainable_pairs.items():
            pending_trainable_pairs[env_aid] = ptup[0]
    else:
        coordinator.gen_simulation_task(task_request, pending_matches)

    if coordinator.task_cache.get(task_request.state_id) is None:
        coordinator.task_cache[task_request.state_id] = TaskCache(
            mode=task_request.computing_mode,
        )
    task_cache = coordinator.task_cache[task_request.state_id]
    if task_cache.n_pending_training_task > 0:
        task_cache.n_pending_training_task -= 1

    task_cache.extend_matches(pending_matches)
    task_cache.update_trainable_pairs(pending_trainable_pairs)
    task_cache.update_population_mapping(pending_population)


@task_handler_register(cls=CoordinatorServer, link=TaskType.UPDATE_PAYOFFTABLE.value)
def _request_update_payoff_table(
    coordinator: CoordinatorServer, task_request: TaskRequest
):
    """Request to update payoff table with local evaluation results. In sync mode, payoff table will be updated until
    all joint policy item have been finished.
    """

    rollout_feedback = task_request.content
    task_cache = coordinator.task_cache[task_request.state_id]
    coordinator.payoff_manager.update_payoff(rollout_feedback)
    if not task_cache.all_training_done():
        return

    population_mapping = task_cache.get_population_mapping()

    all_done = (
        coordinator.payoff_manager.check_done(population_mapping)
        if coordinator.task_mode == "gt"
        else True
    )

    if all_done:
        Logger.debug(
            "* all pending task related to state={} have been updated {}".format(
                task_request.state_id, task_cache.population_mapping
            )
        )
        # update population mapping with trainable policy pair
        trainable_pairs = task_cache.get_trainable_pairs()
        for aid, pid in trainable_pairs.items():
            if pid in population_mapping[aid]:
                continue
            population_mapping[aid].append(pid)

        if len(population_mapping) < 2:
            equilibrium = {
                agent: dict.fromkeys(pm, 1 / len(pm))
                for agent, pm in population_mapping.items()
            }
        else:
            equilibrium = coordinator.payoff_manager.compute_equilibrium(
                population_mapping
            )
        coordinator.payoff_manager.update_equilibrium(population_mapping, equilibrium)

        evaluate_result = coordinator._hyper_evaluator.evaluate(
            task_request.content,
            trainable_mapping=trainable_pairs,
        )

        # clean cache
        task_cache.reset_matches()
        # reset population mapping
        coordinator.task_cache[task_request.state_id].update_population_mapping(
            population_mapping
        )

        if evaluate_result[EvaluateResult.CONVERGED]:
            coordinator._terminate = True
        else:
            task_cache.reset_trainable_pairs()
            for aid in coordinator.training_manager.groups:
                task_cache.n_pending_training_task += 1
                coordinator.gen_add_policy_task(aid, task_request)


@task_handler_register(cls=CoordinatorServer, link=TaskType.ROLLOUT.value)
def _request_rollout(coordinator: CoordinatorServer, task_request: TaskRequest):
    task_request = coordinator.training_manager.retrieve_information(task_request)
    assert isinstance(task_request.content, TrainingFeedback)

    populations = task_request.content.agent_involve_info.populations
    trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
    # then udpate task cache
    if coordinator.task_cache.get(task_request.state_id) is None:
        coordinator.task_cache[task_request.state_id] = TaskCache()
        Logger.debug(
            "generate task cache for state_id={}".format(task_request.state_id)
        )
    coordinator.task_cache[task_request.state_id].update_trainable_pairs(
        {aid: pid for aid, (pid, _) in trainable_pairs.items()}
    )
    population_mapping = {}
    for k, v in populations.items():
        assert len(v) > 0, k
        population_mapping[k] = [p[0] for p in v]
    agent_involve_info = task_request.content.agent_involve_info
    coordinator.task_cache[task_request.state_id].update_population_mapping(
        population_mapping
    )

    if all([len(p_list) for p_list in population_mapping.values()]):
        Logger.debug("rollout with mapping: {}".format(population_mapping))
        policy_distribution = (
            coordinator.payoff_manager.get_equilibrium(population_mapping)
            if coordinator.task_mode == "gt"
            else {
                k: dict(zip(v, [1 / len(v)] * len(v)))
                for k, v in population_mapping.items()
            }
        )
        for env_aid, (pid, _) in agent_involve_info.trainable_pairs.items():
            policy_distribution[env_aid] = {pid: 1.0}
    else:
        policy_distribution = {}

    rollout_config = coordinator._configs["rollout"]
    task_config = rollout_config["task_config"]
    task = TaskDescription(
        task_type=TaskType.ROLLOUT,
        content=RolloutDescription(
            agent_involve_info=agent_involve_info,
            fragment_length=task_config["fragment_length"],
            max_step=task_config["max_step"],
            callback=rollout_config["callback"],
            stopper=rollout_config["stopper"]["name"],
            stopper_config=rollout_config["stopper"]["config"],
            policy_distribution=policy_distribution,
        ),
        state_id=task_request.state_id,
    )

    coordinator.rollout_manager.rollout(task_desc=task)


__all__ = ["CoordinatorServer"]
