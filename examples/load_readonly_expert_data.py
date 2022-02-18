# # -*- encoding: utf-8 -*-
# # -----
# # Created Date: 2021/7/16
# # Author: Hanjing Wang
# # -----
# # Last Modified:
# # Modified By:
# # -----
# # Copyright (c) 2020 MARL @ SJTU
# # -----

# import argparse

# import yaml
# import os

# from malib.envs import MPE
# from malib.runner import run


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("General training on Atari games.")
#     parser.add_argument(
#         "--config", type=str, help="YAML configuration path.", required=True
#     )

#     args = parser.parse_args()

#     with open(os.path.join(BASE_DIR, args.config), "r") as f:
#         config = yaml.load(f)

#     env_desc = config["env_description"]
#     env_desc["config"] = env_desc.get("config", {})
#     # load creator
#     env_desc["creator"] = MPE
#     env = MPE(**env_desc["config"])

#     possible_agents = env.possible_agents
#     observation_spaces = env.observation_spaces
#     action_spaces = env.action_spaces

#     env_desc["possible_agents"] = env.possible_agents
#     env.close()

#     training_config = config["training"]
#     rollout_config = config["rollout"]

#     training_config["interface"]["observation_spaces"] = observation_spaces
#     training_config["interface"]["action_spaces"] = action_spaces

#     dataset_config = config.get("dataset_config", {})
#     dataset_config.update(
#         {
#             "episode_capacity": 10000,
#             # Define the starting job,
#             # currently only support data loading
#             "init_job": {
#                 # main dataset loads the data
#                 "load_when_start": False,
#                 "path": "/home/hanjing/malib-custom/data/",
#             },
#             # Configuration for external resources,
#             # possibly multiple dataset shareds with
#             # extensive optimization like load balancing,
#             # heterogeneously organized data and etc.
#             # currently only support Read-Only dataset
#             # Datasets and returned data are organized as:
#             # ----------------       --------------------
#             # | Main Dataset | ---  | External Dataset [i] |
#             # ---------------       --------------------
#             "extern": {
#                 "links": [
#                     {
#                         "name": "expert data",
#                         "path": "data/expert/",
#                         "write": False,
#                     }
#                 ],
#                 "sample_rates": [1],
#             },
#             # Define the quitting job here,
#             # Again, only dumping is supported now.
#             "quit_job": {
#                 "dump_when_closed": True,
#                 "path": "/home/hanjing/malib-custom/data/",
#             },
#         }
#     )
#     parameter_server_config = config.get("parameter_server", {})
#     parameter_server_config.update(
#         {
#             "init_job": {
#                 "load_when_start": False,
#                 "path": "/home/hanjing/malib-custom/checkpoints/",
#             },
#             "quit_job": {
#                 "dump_when_closed": True,
#                 "path": "/home/hanjing/malib-custom/checkpoints/",
#             },
#         }
#     )
#     run(
#         group=config["group"],
#         name=config["name"],
#         env_description=env_desc,
#         agent_mapping_func=lambda agent: "share",
#         training=training_config,
#         algorithms=config["algorithms"],
#         # rollout configuration for each learned policy model
#         rollout=rollout_config,
#         evaluation=config.get("evaluation", {}),
#         global_evaluator=config["global_evaluator"],
#         dataset_config=dataset_config,
#         parameter_server=parameter_server_config,
#     )
