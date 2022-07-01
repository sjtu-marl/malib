# from malib.algorithm.mappo import MAPPO
# from yaml import load, dump
# from test_da import FakeDataServer

# from malib.rollout.envs.agent_interface import AgentInterface
# from malib.rollout.rollout_func import env_runner
# from malib.utils.typing import BufferDescription

# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# import pytest

# from malib.rollout.envs.gr_football import creator
# from pathlib import Path
# from malib.rollout.envs.vector_env import VectorEnv
# import ray


# @pytest.mark.parametrize(
#     "env_name,n_player_left,n_player_right",
#     [("5_vs_5", 4, 0)],
# )
# class TestMAPPOonFootball:
#     @pytest.fixture(autouse=True)
#     def _init(
#         self,
#         env_name,
#         n_player_left: int,
#         n_player_right: int,
#     ):
#         if not ray.is_initialized():
#             ray.init(local_mode=True)

#         scenario_configs = {
#             "env_name": env_name,
#             "number_of_left_players_agent_controls": n_player_left,
#             "number_of_right_players_agent_controls": n_player_right,
#             "representation": "raw",
#             "logdir": "",
#             "write_goal_dumps": False,
#             "write_full_episode_dumps": False,
#             "render": False,
#             "stacked": False,
#         }

#         self.env = creator(env_id="PSFootball", scenario_configs=scenario_configs)
#         self.env_creator = creator
#         self.env_id = "PSFootball"
#         self.scenario_configs = scenario_configs
#         self.build_policy()

#     def build_policy(self):
#         base_dir = Path(__file__).parent
#         cfg = load(open(base_dir / "mappo_5_vs_5.yaml", "r"))
#         algo_cfg = cfg["algorithms"]["MAPPO"]
#         model_cfg = algo_cfg["model_config"]
#         custom_cfg = algo_cfg["custom_config"]

#         custom_cfg.update({"global_state_space": self.env.state_spaces})

#         self.policies = {
#             aid: MAPPO(
#                 "MAPPO",
#                 self.env.observation_spaces[aid],
#                 self.env.action_spaces[aid],
#                 model_cfg,
#                 custom_cfg,
#                 env_agent_id=aid,
#             )
#             for aid in self.env.possible_agents
#         }

#     def test_rollout(self):
#         vec_env = VectorEnv(
#             self.env.observation_spaces,
#             self.env.action_spaces,
#             self.env_creator,
#             configs={"scenario_configs": self.scenario_configs, "env_id": self.env_id},
#         )

#         agent_interfaces = {
#             aid: AgentInterface(
#                 aid,
#                 self.env.observation_spaces[aid],
#                 self.env.action_spaces[aid],
#                 parameter_server=None,
#                 policies={"1": self.policies[aid]},
#             )
#             for aid in self.env.possible_agents
#         }

#         vec_env.add_envs(num=4)

#         _ = [interface.reset() for interface in agent_interfaces.values()]

#         behavior_policy_ids = {
#             agent_id: interface.behavior_policy
#             for agent_id, interface in agent_interfaces.items()
#         }
#         dataset = FakeDataServer.remote()
#         buffer_desc = BufferDescription(
#             env_id=vec_env.env_configs["env_id"],
#             agent_id=vec_env.possible_agents,
#             policy_id=[behavior_policy_ids[aid] for aid in vec_env.possible_agents],
#         )

#         rollout_info = env_runner(
#             vec_env,
#             agent_interfaces,
#             buffer_desc=buffer_desc,
#             runtime_config={
#                 "max_step": 3001,
#                 "num_envs": 1,
#                 "fragment_length": 3001,
#                 "behavior_policies": behavior_policy_ids,
#                 "custom_reset_config": None,
#                 "batch_mode": "episode",
#                 "postprocessor_type": "value",
#             },
#             dataset_server=dataset,
#         )
#         print(rollout_info)
