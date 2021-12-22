# # -*- coding:utf-8 -*-
# # Create Date: 12/11/20, 2:46 PM
# # Author: ming
# # ---
# # Last Modified: 12/11/20
# # Modified By: ming
# # ---
# # Copyright (c) 2020 MARL @ SJTU

# import time
# import logging
# import ray
# import traceback

# from typing import Dict, Any, Callable, Tuple, Sequence

# from malib.policy.meta_policy import MetaPolicy
# from malib.utils.formatter import pretty_print as pp
# from malib.agent.agent import Agent
# from malib.utils.logger import get_logger
# from malib.utils.typing import (
#     PolicyID,
#     AgentID,
#     TaskType,
#     TaskRequest,
#     Status,
#     TrainingFeedback,
#     BColors,
#     Paradigm,
#     TaskDescription,
#     AgentInvolveInfo,
#     MetaParameterDescription,
# )


# @ray.remote
# class TrainingManager:
#     """Async multi-agent training manager"""

#     def __init__(
#         self,
#         paradigm: Paradigm,
#         env_desc: Dict[str, Any],
#         coordinator: str,
#         meta_policy_mapping_func: Callable,
#         agent_configs: Dict[AgentID, Dict[str, Any]],
#         log_dir: str,
#         log_level=logging.DEBUG,
#     ):
#         """Create an TrainingManager instance which is responsible for the multi agent training
#         tasks execution and rollout task requests sending.

#         Parameters
#         ----------
#         paradigm
#             Paradigm, the specified paradigm, should be MARL or Meta Game, which determines the training manner.
#         env_desc
#             Dict[str, Any], the description for environment generation.
#         coordinator
#             str, the actor name used to build connection with a remote Ray-based coordinator actor.
#         meta_policy_mapping_func.
#             Callable, the mapping function maps agent id to meta policy id.
#         agent_configs
#             Dict[AgentID, Dict[str, Any]], the agent configuration dictionary.
#         log_dir
#             str, the logging root directory.
#         log_level
#             int, the logging level, default by logging.DEBUG
#         """

#         self._logger = get_logger(log_level, log_dir, "training_manager.py")

#         self._paradigm = paradigm
#         self._env_description = env_desc
#         self._agent_configs = agent_configs
#         self._meta_policy_mapping_func = meta_policy_mapping_func

#         self._mpid_to_aid = {
#             self._meta_policy_mapping_func(aid): aid for aid in agent_configs.keys()
#         }

#         # XXX(ming): Try to build connection with coordinator server. We just retry without
#         #  failure limits currently, but it is not a proper way, should limit the retries with
#         #  setting max failures in the future work.
#         while True:
#             try:
#                 self._logger.info("Try to connect to coordinator server ...")
#                 self._coordinator = ray.get_actor(coordinator)
#                 self._logger.info("Connected to coordinator server")
#                 break
#             except Exception as e:
#                 self._logger.warning(f"Waiting for coordinator server... {e}")
#                 time.sleep(2)
#                 continue

#         self._agents = {
#             aid: Agent.remote(
#                 aid,
#                 self._meta_policy_mapping_func(aid),
#                 self._env_description["id"],
#                 coordinator,
#                 log_level,
#                 log_dir,
#                 *configs,
#             )
#             for aid, configs in self._agent_configs.items()
#         }

#         # XXX(ming): is it necessary to maintain a local population?
#         trainable_pairs = {}
#         for aid, agent in self._agents.items():
#             _, pid, description = ray.get(agent.add_policy.remote(None))
#             trainable_pairs[self._meta_policy_mapping_func(aid)] = (pid, description)
#         self._logger.debug(f"{len(self._agents)} agents have been created")

#         # no matter what the paradigm is, we
#         self.send(TaskType.INIT, trainable_pairs=trainable_pairs, statistics={})

#         self._pending_task_ids = []

#     def run(self):
#         self._logger.debug("waiting for training tasks ...")
#         while True:
#             try:
#                 task = ray.get(self._coordinator.get_training_task.remote())
#                 if task.status == Status.TERMINATE:
#                     self._logger.info("Will close the TrainingManager server.")
#                     break
#                 self.execute(task)
#             except Exception as e:
#                 self._logger.error(BColors.FAIL + traceback.format_exc() + BColors.ENDC)

#     def execute(self, task: TaskDescription):
#         """Execute specific task.

#         Legal tasks include `SAVE_MODEL`, `OPTIMIZE`, `LOAD_MODEL` and `ADD_POLICY`.

#         Returns
#         -------
#         tuple:
#             return a tuple of task type and feedback.
#         """

#         if task.task_type == TaskType.SAVE_MODEL:
#             task_type, feedback = self.save(task)
#         elif task.task_type == TaskType.OPTIMIZE:
#             task_type, feedback = self.train(task)
#         elif task.task_type == TaskType.LOAD_MODEL:
#             task_type, feedback = self.load(task)
#         elif task.task_type == TaskType.ADD_POLICY:
#             task_type, feedback = self.add_policy(task)
#         elif task.task_type == TaskType.CHECK_ADD:
#             task_type, feedback = None, None
#         else:
#             raise ValueError(f"Unexpected task type: {task.task_type}")

#         if task_type is not None:
#             self.send(task_type, **feedback)
#         else:
#             # XXX(ming): currently, we cannot perform it in async mode, will stuck
#             while len(self._pending_task_ids) > 0:
#                 done_task_ids, self._pending_task_ids = ray.wait(self._pending_task_ids)
#                 for done_id in done_task_ids:
#                     task_type, raw_feed_back = ray.get(done_id)
#                     self.send(task_type, **raw_feed_back)

#     def _get_population_desc(self) -> Dict[PolicyID, Sequence[Any]]:
#         """Return stationary population description"""

#         mpid_pdesc_tup_ids = [
#             agent.get_stationary_descriptions.remote()
#             for agent in self._agents.values()
#         ]
#         mpid_pdesc_tup = ray.get(mpid_pdesc_tup_ids)
#         populations = dict(mpid_pdesc_tup)

#         return populations

#     def add_policy(self, task: TaskDescription) -> Tuple[Any, Any]:
#         """Generate new policies according to the trainable pairs of the given TaskDescription.

#         Parameters
#         ----------
#         task
#             TaskDescription, the TaskDescription instance.

#         Returns
#         -------
#         Tuple
#             a tuple of None
#         """

#         content = task.training_task.agent_involve_info
#         trainable_pairs = {}  # mapping from meta-policy ids to tuples

#         # should get a trainable pairs which contains all agents
#         self._logger.debug(
#             f"add policies for all agents with trainable pairs in expected "
#             f"(source_task_id={task.source_task_id}):\n{content.trainable_pairs}"
#         )
#         # assert list(self._agents.keys()) == list(
#         #     map(lambda x: self._mpid_to_aid[x], content.trainable_pairs.keys())
#         # ), (list(self._agents.keys()), list(content.trainable_pairs.keys()))

#         pending_task_ids = [
#             self._agents[self._mpid_to_aid[mpid]].add_policy.remote(None)
#             for mpid, _ in content.trainable_pairs.items()
#         ]
#         # wait until all involved agents generated new policies.
#         while len(pending_task_ids):
#             done_task_ids, pending_task_ids = ray.wait(pending_task_ids)
#             for done_id in done_task_ids:
#                 agent_id, policy_id, policy_description = ray.get(done_id)
#                 trainable_pairs[self._meta_policy_mapping_func(agent_id)] = (
#                     policy_id,
#                     policy_description,
#                 )

#         self._logger.debug(f"generated new policies: {pp(trainable_pairs, 1)}")

#         # retrieve agent meta parameter seq
#         task_ids = [
#             agent.require_parameter_desc.remote() for agent in self._agents.values()
#         ]
#         meta_parameter_desc_seq: Sequence[MetaParameterDescription] = ray.get(task_ids)

#         # mapping from meta-policy id to MetaParameterDescription
#         meta_parameter_dict = {
#             meta_parameter_desc.id: meta_parameter_desc
#             for meta_parameter_desc in meta_parameter_desc_seq
#         }

#         if self._paradigm == Paradigm.META_GAME:
#             ########################################################################
#             # Caution: currently only for poker, if other env please comment it!
#             ########################################################################
#             # from malib.gt.algorithms.exploitability import measure_exploitability
#             # meta_policies = dict()
#             # for agent in self._agents.values():
#             #     mpid, population_descs = ray.get(
#             #         agent.get_stationary_descriptions.remote()
#             #     )
#             #     meta_policies[mpid] = MetaPolicy(
#             #         mpid,
#             #         population_descs[0][1]["observation_space"],
#             #         population_descs[0][1]["action_space"],
#             #     )
#             #     for pid, pconfig in population_descs:
#             #         meta_policies[mpid].add_policy(pid, pconfig)
#             #         param_desc = meta_parameter_dict[mpid].parameter_desc_dict[pid]
#             #         parameter = ray.get(
#             #             self._coordinator.pull_parameter.remote(param_desc),
#             #             timeout=10,
#             #         )
#             #
#             #         meta_policies[mpid].population[param_desc.id].set_weights(parameter)
#             #
#             # exploitability, expl_per_player = measure_exploitability(
#             #     malib_meta_policies=meta_policies,
#             #     poker_game_name="leduc_poker",
#             #     policy_mixture_dict=task.training_task.policy_distribution,
#             # )
#             # self._logger.info(
#             #     f"NE: {pp(task.training_task.policy_distribution, 1)}"
#             #     f"Exploitability: {exploitability}, Exploitability per player: {expl_per_player}"
#             # )

#             for mpid, p_tuple in trainable_pairs.items():
#                 feedback = {
#                     "trainable_pairs": {mpid: p_tuple},
#                     "statistics": {},
#                     "meta_parameter_desc_dict": meta_parameter_dict,
#                 }
#                 self._pending_task_ids.append(ray.put((TaskType.ROLLOUT, feedback)))
#                 self._logger.debug(f"Put rollout task:\n\t{feedback}")
#         else:
#             raise ValueError("Not supported paradigm")

#         return None, None

#     def train(self, task_desc: TaskDescription) -> Tuple[Any, Any]:
#         """Execute training tasks.

#         Notes
#         -----
#         The end conditions of training task determined by locally convergence judgement or...

#         Parameters
#         ----------
#         task_desc

#         Returns
#         -------

#         """
#         self._logger.debug(
#             f"Receive a TRAINING task:\n\t{task_desc.training_task.agent_involve_info}"
#         )
#         training_task = task_desc.training_task
#         for mpid, (pid, _) in training_task.agent_involve_info.trainable_pairs.items():
#             aid = self._mpid_to_aid[mpid]
#             agent = self._agents[aid]
#             self._pending_task_ids.append(
#                 agent.train.remote(
#                     policy_id=pid,
#                     batch_size=training_task.batch_size,
#                     sample_mode=training_task.mode,
#                     num_epoch=training_task.num_epoch,
#                 )
#             )

#         if self._paradigm == Paradigm.MARL:
#             self._logger.warning("may be wrong")
#         else:
#             pass
#         return None, None

#     def send(
#         self,
#         task_type: TaskType,
#         *,
#         trainable_pairs: Dict[PolicyID, Tuple],
#         statistics,
#         meta_parameter_desc_dict=None,
#     ):
#         """Send callback task requests to coordinator."""

#         populations = self._get_population_desc()
#         if meta_parameter_desc_dict is None:
#             task_ids = [
#                 agent.require_parameter_desc.remote() for agent in self._agents.values()
#             ]
#             meta_parameter_desc_seq: Sequence[MetaParameterDescription] = ray.get(
#                 task_ids
#             )
#             meta_parameter_desc_dict = {
#                 meta_parameter_desc.id: meta_parameter_desc
#                 for meta_parameter_desc in meta_parameter_desc_seq
#             }
#         task_request = TaskRequest(
#             task_type=task_type,
#             status=Status.NORMAL,
#             content=TrainingFeedback(
#                 agent_involve_info=AgentInvolveInfo(
#                     env_id=self._env_description["id"],
#                     populations=populations,
#                     trainable_pairs=trainable_pairs,
#                     meta_parameter_desc_dict=meta_parameter_desc_dict,
#                 ),
#                 statistics=statistics,
#             ),
#         )
#         self._coordinator.request.remote(task_request)

#     @ray.method(num_returns=1)
#     def save(self, task_desc: TaskDescription):
#         info = ""
#         return (TaskType.NO, info)

#     @ray.method(num_returns=1)
#     def load(self, task_desc: TaskDescription):
#         info = ""
#         return (TaskType.NO, info)
