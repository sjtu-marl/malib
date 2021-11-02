from os import link
import threading

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
)
from malib.utils.tasks_register import task_handler_register, helper_register
from malib.backend.coordinator.server import CoordinatorServer


@helper_register(cls=CoordinatorServer)
def gen_simulation_task(
    self: CoordinatorServer, task_request: TaskRequest, matches: List
):
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


@helper_register(cls=CoordinatorServer)
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


@task_handler_register(cls=CoordinatorServer, link=TaskType.OPTIMIZE)
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


@task_handler_register(cls=CoordinatorServer, link=TaskType.SIMULATION)
def _request_simulation(coordinator: CoordinatorServer, task_request: TaskRequest):

    Logger.debug("request for simulation")
    # fill message for this request
    task_request = coordinator.training_manger.retrieve_information(task_request)

    # generate pending matches
    pending_matches = []
    pending_trainable_pairs = {}
    # cache task related information
    coordinator.task_cache[task_request.state_id]["matches"] = pending_matches
    coordinator.task_cache[task_request.state_id][
        "trainable_pairs"
    ] = pending_trainable_pairs
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


@task_handler_register(cls=CoordinatorServer, link=TaskType.EVALUATE)
def _request_evaluation(coordinator: CoordinatorServer, task_request: TaskRequest):
    # TODO(ming): add population mapping description
    Logger.debug("rollout done, request for evaluation")
    trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
    pending_matches = []
    pending_trainable_pairs = {}

    coordinator.task_cache[task_request.state_id]["matches"] = pending_matches
    coordinator.task_cache[task_request.state_id][
        "trainable_pairs"
    ] = pending_trainable_pairs
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


@task_handler_register(cls=CoordinatorServer, link=TaskType.UPDATE_PAYOFFTABLE)
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
        all_done = coordinator.payoff_manager.check_done(
            task_cache["population_mapping"]
        )
        if all_done:
            Logger.debug(
                "all pending task related to state={} have been updated".format(
                    task_request.state_id
                )
            )
            # update population mapping with trainable policy pair
            population_mapping = task_cache["population_mapping"]
            trainable_pairs = task_cache["trainable_pairs"]
            for aid, pid in trainable_pairs:
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

            coordinator.update_equilibrium(population_mapping, equilibrium)

            evaluate_result = coordinator._hyper_evaluator.evaluate(
                # content here should be
                task_request.content,
                # weighted_payoffs=weighted_payoffs,
                # oracle_payoffs=oracle_payoffs,
                trainable_mapping=task_cache["trainable_pairs"],
            )
            if evaluate_result[EvaluateResult.CONVERGED]:
                coordinator._terminate = True
            else:
                coordinator.task_cache[task_request.state_id]["trainable_pairs"] = {}
                for aid in coordinator.training_manager.groups:
                    coordinator.gen_add_policy_task(aid, task_request.state_id)

            # clean cache
            coordinator.task_cache[task_request.state_id]["matches"] = None
            coordinator.task_cache[task_request.state_id]["trainable_pairs"] = None
        else:
            Logger.warning(
                "state={} is waiting for other sub tasks ...".format(
                    task_request.state_id
                )
            )


@task_handler_register(cls=CoordinatorServer, link=TaskType.ROLLOUT)
def _request_rollout(coordinator: CoordinatorServer, task_request: TaskRequest):
    task_request = coordinator.training_manager.retrieve_information(task_request)
    assert isinstance(task_request.content, TrainingFeedback)

    populations = task_request.content.agent_involve_info.populations
    population_mapping = {}
    for k, v in populations.items():
        assert len(v) > 0, v
        population_mapping[k] = [p[0] for p in v]
    agent_involve_info = task_request.content.agent_involve_info

    if all([len(p_list) for p_list in population_mapping.values()]):
        policy_distribution = coordinator.payoff_manager.get_equilibrium(
            population_mapping
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
        state_id=None,
    )

    coordinator.rollout_manager.rollout(task_desc=task)
