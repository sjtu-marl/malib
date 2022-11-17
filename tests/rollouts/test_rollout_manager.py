# # MIT License

# # Copyright (c) 2021 MARL @ SJTU

# # Author: Ming Zhou

# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# from typing import Sequence, Dict, Any, Callable, List

# import pytest
# import ray

# from ray.util import ActorPool

# from malib.utils.typing import AgentID
# from malib.common.strategy_spec import StrategySpec
# from malib.rollout.rolloutworker import RolloutWorker
# from malib.rollout.manager import RolloutWorkerManager


# class FakeRolloutWorker(RolloutWorker):
#     def init_agent_interfaces(
#         self, env_desc: Dict[str, Any], runtime_ids: Sequence[AgentID]
#     ) -> Dict[AgentID, Any]:
#         return {}

#     def init_actor_pool(
#         self,
#         env_desc: Dict[str, Any],
#         rollout_config: Dict[str, Any],
#         agent_mapping_func: Callable,
#     ) -> ActorPool:
#         return NotImplementedError

#     def init_servers(self):
#         pass

#     def rollout(
#         self,
#         runtime_strategy_specs: Dict[str, StrategySpec],
#         stopping_conditions: Dict[str, Any],
#         data_entrypoints: Dict[str, str],
#         trainable_agents: List[AgentID] = None,
#     ):
#         self.set_running(True)
#         return {}

#     def simulate(self, runtime_strategy_specs_list: List[Dict[str, StrategySpec]]):
#         raise NotImplementedError

#     def step_rollout(
#         self,
#         eval_step: bool,
#         rollout_config: Dict[str, Any],
#         dataset_writer_info_dict: Dict[str, Any],
#     ) -> List[Dict[str, Any]]:
#         pass

#     def step_simulation(
#         self,
#         runtime_strategy_specs_list: List[Dict[str, StrategySpec]],
#         rollout_config: Dict[str, Any],
#     ) -> List[Dict[str, Any]]:
#         pass


# def create_manager(
#     stopping_conditions: Dict[str, Any],
#     rollout_config: Dict[str, Any],
#     env_desc: Dict[str, Any],
# ):
#     return RolloutWorkerManager(
#         experiment_tag="test_rollout_manager",
#         stopping_conditions=stopping_conditions,
#         agent_mapping_func=lambda agent: agent,
#         rollout_config=rollout_config,
#         env_desc=env_desc,
#         log_dir="./logs",
#         rollout_worker_cls=FakeRolloutWorker,
#     )


# @pytest.mark.parametrize("n_players", [1, 2, 4])
# class TestRolloutManager:
#     def test_rollout_task_send(self, n_players: int):
#         if not ray.is_initialized():
#             ray.init()

#         agents = [f"player_{i}" for i in range(n_players)]
#         manager = create_manager(
#             stopping_conditions={"rollout": {}},
#             rollout_config=None,
#             env_desc={"possible_agents": agents},
#         )

#         task_list = [
#             {"trainable_agents": None, "data_entrypoints": None} for _ in range(3)
#         ]
#         manager.rollout(task_list)
#         ray.shutdown()

#     # def test_simulation_task_send(self, n_players: int):
#     #     if not ray.is_initialized():
#     #         ray.init()

#     #     agents = [f"player_{i}" for i in range(n_players)]
#     #     manager = create_manager(
#     #         stopping_conditions={"rollout": {}},
#     #         rollout_config=None,
#     #         env_desc={"possible_agents": agents},
#     #     )

#     #     manager.rollout(task_list)
#     #     ray.shutdown()
