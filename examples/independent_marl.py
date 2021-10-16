# """
# Implementation of independent learning applied to MARL cases.
# """

# import argparse

# from malib.envs.mpe.env import MPE
# from malib.runner import run

# parser = argparse.ArgumentParser(
#     "Independent multi-agent learning on mpe environments."
# )

# parser.add_argument("--batch_size", type=int, default=64)
# parser.add_argument("--num_epoch", type=int, default=100)
# parser.add_argument("--fragment_length", type=int, default=25)
# parser.add_argument("--worker_num", type=int, default=6)
# parser.add_argument("--algorithm", type=str, default="PPO")


# if __name__ == "__main__":
#     args = parser.parse_args()
#     env_description = {
#         "creator": MPE,
#         "config": {
#             "env_id": "simple_tag_v2",
#             "scenario_configs": {
#                 "num_good": 2,
#                 "num_adversaries": 2,
#                 "num_obstacles": 2,
#                 "max_cycles": 25,
#             },
#         },
#     }
#     env = MPE(**env_description["config"])
#     env_description["possible_agents"] = env.possible_agents

#     run(
#         env_description=env_description,
#         training={
#             "interface": {
#                 "type": "independent",
#                 "observation_spaces": env.observation_spaces,
#                 "action_spaces": env.action_spaces,
#             },
#             "config": {
#                 "agent": {
#                     "observation_spaces": env.observation_spaces,
#                     "action_spaces": env.action_spaces,
#                 },
#                 "batch_size": args.batch_size,
#                 "grad_norm_clipping": 0.5,
#             },
#         },
#         algorithms={"PPO": {"name": "PPO"}},
#         rollout={
#             "type": "async",
#             "stopper": "simple_rollout",
#             "metric_type": "simple",
#             "fragment_length": 75,
#             "num_episodes": 100,
#         },
#         global_evaluator={
#             "name": "generic",
#             "config": {"stop_metrics": {}},
#         },
#     )
