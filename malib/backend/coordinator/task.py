import threading

from dataclasses import dataclass, field

from malib.utils.logger import Logger
from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    TaskType,
    TrainingFeedback,
    TrainingDescription,
    RolloutDescription,
    AgentInvolveInfo,
    SimulationDescription,
    List,
    EvaluateResult,
    Dict,
)
from malib.utils.tasks_register import task_handler_register, helper_register
from malib.backend.coordinator.base_coordinator import BaseCoordinator
from malib.backend.coordinator.server import CoordinatorServer


@dataclass
class TaskCache:
    matches: List = field(default_factory=list)
    trainable_pairs: Dict = field(default_factory=dict)
    population_mapping: Dict = field(default_factory=dict)

    def clean(self):
        self.matches = []
        self.trainable_pairs = {}
        self.population_mapping = None


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
    # XXX(ming): may we can remove it
    with threading.Lock():
        task_request = coordinator.training_manager.retrieve_information(task_request)

    # generate pending matches
    pending_matches = []
    pending_trainable_pairs = {}

    # cache task related information
    for (
        env_aid,
        p_tup,
    ) in task_request.content.agent_involve_info.trainable_pairs.items():
        pending_trainable_pairs[env_aid] = p_tup[0]
        pending_matches.extend(
            coordinator.payoff_manager.get_pending_matchups(env_aid, *p_tup)
        )
    if len(pending_matches) > 0:
        coordinator.gen_simulation_task(task_request, pending_matches)

    with threading.Lock():
        if coordinator.task_cache.get(task_request.state_id) is None:
            coordinator.task_cache[task_request.state_id] = TaskCache()
            Logger.debug(
                "generate task cache for state_id={}".format(task_request.state_id)
            )
        coordinator.task_cache[task_request.state_id].matches.extend(pending_matches)
        coordinator.task_cache[task_request.state_id].trainable_pairs.update(
            pending_trainable_pairs
        )
        coordinator.task_cache[task_request.state_id].population_mapping = {
            aid: [e[0] for e in p_tup]
            for aid, p_tup in task_request.content.agent_involve_info.populations.items()
        }


@task_handler_register(cls=CoordinatorServer, link=TaskType.EVALUATE.value)
def _request_evaluation(coordinator: CoordinatorServer, task_request: TaskRequest):
    # TODO(ming): add population mapping description
    Logger.debug("rollout done, request for evaluation")
    trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
    pending_matches = []
    pending_trainable_pairs = {}

    for env_aid, ptup in trainable_pairs.items():
        pending_matches.extend(
            coordinator.payoff_manager.get_pending_matchups(env_aid, *ptup)
        )

    if len(pending_matches) == 0:
        Logger.warning("repeated policy id detected!")
        for env_aid, ptup in trainable_pairs.items():
            pending_trainable_pairs[env_aid] = ptup
    else:
        coordinator.gen_simulation_task(task_request, pending_matches)

    with threading.Lock():
        if coordinator.task_cache.get(task_request.state_id) is None:
            coordinator.task_cache[task_request.state_id] = TaskCache()
        coordinator.task_cache[task_request.state_id].matches.extend(pending_matches)
        coordinator.task_cache[task_request.state_id].trainable_pairs.update(
            pending_trainable_pairs
        )
        coordinator.task_cache[task_request.state_id].population_mapping = {
            aid: [e[0] for e in p_tup]
            for aid, p_tup in task_request.content.agent_involve_info.populations.items()
        }


@task_handler_register(cls=CoordinatorServer, link=TaskType.UPDATE_PAYOFFTABLE.value)
def _request_update_payoff_table(
    coordinator: CoordinatorServer, task_request: TaskRequest
):
    """Request to update payoff table with local evaluation results. In sync mode, payoff table will be updated until
    all joint policy item have been finished.
    """

    rollout_feedback = task_request.content
    with threading.Lock():
        coordinator.payoff_manager.update_payoff(rollout_feedback)
        task_cache = coordinator.task_cache[task_request.state_id]
        all_done = (
            coordinator.payoff_manager.check_done(task_cache.population_mapping)
            if coordinator.task_mode == "gt"
            else True
        )
        if all_done:
            Logger.debug(
                "all pending task related to state={} have been updated".format(
                    task_request.state_id
                )
            )
            # update population mapping with trainable policy pair
            population_mapping = task_cache.population_mapping
            trainable_pairs = task_cache.trainable_pairs
            for aid, pid in trainable_pairs.items():
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

            coordinator.payoff_manager.update_equilibrium(
                population_mapping, equilibrium
            )

            evaluate_result = coordinator._hyper_evaluator.evaluate(
                # content here should be
                task_request.content,
                # weighted_payoffs=weighted_payoffs,
                # oracle_payoffs=oracle_payoffs,
                trainable_mapping=task_cache.trainable_pairs,
            )
            if evaluate_result[EvaluateResult.CONVERGED]:
                coordinator._terminate = True
            else:
                coordinator.task_cache[task_request.state_id].trainable_pairs = {}
                for aid in coordinator.training_manager.groups:
                    coordinator.gen_add_policy_task(aid, task_request.state_id)

            # clean cache
            with threading.Lock():
                coordinator.task_cache[task_request.state_id].clean()
                # reset population mapping
                coordinator.task_cache[
                    task_request.state_id
                ].population_mapping = population_mapping
        else:
            Logger.warning(
                "state={} is waiting for other sub tasks ... with population mapping: {}".format(
                    task_request.state_id, task_cache.population_mapping
                )
            )


@task_handler_register(cls=CoordinatorServer, link=TaskType.ROLLOUT.value)
def _request_rollout(coordinator: CoordinatorServer, task_request: TaskRequest):
    task_request = coordinator.training_manager.retrieve_information(task_request)
    assert isinstance(task_request.content, TrainingFeedback)

    populations = task_request.content.agent_involve_info.populations
    population_mapping = {}
    for k, v in populations.items():
        # FIXME(ming): sometimes may no population
        assert len(v) > 0, k
        population_mapping[k] = [p[0] for p in v]
    agent_involve_info = task_request.content.agent_involve_info

    if all([len(p_list) for p_list in population_mapping.values()]):
        policy_distribution = (
            coordinator.payoff_manager.get_equilibrium(population_mapping)
            if coordinator.task_mode == "gta"
            else {
                k: dict(zip(v, [1 / len(v)] * len(v)))
                for k, v in population_mapping.items()
            }
        )
        for env_aid, (pid, _) in agent_involve_info.trainable_pairs.items():
            policy_distribution[env_aid] = {pid: 1.0}
        # since in meta_policy this is a default_dict with value 0.0
    else:
        policy_distribution = {}

    rollout_config = coordinator._configs["rollout"]
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
        state_id=task_request.state_id,
    )

    coordinator.rollout_manager.rollout(task_desc=task)


__all__ = ["CoordinatorServer"]
