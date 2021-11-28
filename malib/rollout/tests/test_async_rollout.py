# import logging
# from malib.utils.metrics import get_metric
# from sys import intern
# from typing import Sequence
# import gym
# from malib.envs.agent_interface import AgentInterface
# from malib.envs import Environment
# from malib.envs.vector_env import VectorEnv
# from malib.rollout.rollout_func import sequential, simultaneous
# import os
# import pytest
# import yaml

# from malib import settings
# from malib.utils.typing import (
#     BehaviorMode,
#     Dict,
#     AgentID,
#     ObservationSpaceType,
#     PolicyID,
# )
# from malib.algorithm.random.policy import RandomPolicy
# from malib.envs.maatari.env import MAAtari
# from malib.envs.poker import poker_aec_env


# @pytest.fixture(scope="session")
# def poker_config():
#     return {"fixed_player": True}


# def _gen_agent_interfaces(env: Environment):
#     observation_spaces = env.observation_spaces
#     action_spaces = env.action_spaces

#     res = {}
#     for agent in env.possible_agents:
#         random_policy = RandomPolicy(
#             "random", observation_spaces[agent], action_spaces[agent], {}, {}
#         )
#         res[agent] = AgentInterface(
#             agent_id=agent,
#             observation_space=observation_spaces[agent],
#             action_space=action_spaces[agent],
#             parameter_server=None,
#             policies={f"random_{i}": random_policy for i in range(2)},
#         )
#     return res


# def _gen_agent_episodes(
#     trainable_agents: Sequence[AgentID],
#     behavior_policies: Dict[AgentID, PolicyID],
#     env_id: str,
#     capacity: int,
#     other_columns: Sequence[str],
#     is_sequential: bool = False,
# ):

#     res = {}
#     episode_creator = Episode if not is_sequential else SequentialEpisode
#     for agent in trainable_agents:
#         res[agent] = episode_creator(
#             env_id,
#             behavior_policies[agent],
#             capacity=capacity,
#             other_columns=other_columns,
#         )
#     return res


# @pytest.fixture(scope="session")
# def maatari_config():
#     with open(
#         os.path.join(settings.BASE_DIR, "examples/configs/dqn_basketball_pong.yaml"),
#         "r",
#     ) as f:
#         config = yaml.load(f)
#     return config


# def test_vector_rollout(maatari_config):
#     creator = MAAtari
#     configs = maatari_config["env_description"]["config"]
#     rollout_configs = maatari_config["rollout"]
#     env = creator(**configs)

#     # generate agent interfaces
#     agent_interfaces: Dict[AgentID, AgentInterface] = _gen_agent_interfaces(env)
#     _ = [interface.reset() for interface in agent_interfaces.values()]

#     # sample behavior policies
#     behavior_policies = {
#         agent: list(interface.policies.keys())[0]
#         for agent, interface in agent_interfaces.items()
#     }

#     # create 10 environments to form a vectored environment
#     vec_env = VectorEnv.from_envs([env], configs)
#     vec_env.add_envs(num=9)

#     # generate agent episodes
#     agent_episodes = _gen_agent_episodes(
#         env.trainable_agents,
#         behavior_policies,
#         "basket_ball_pong",
#         rollout_configs["fragment_length"] * 2,
#         env.extra_returns,
#     )

#     metric = get_metric(rollout_configs["metric_type"])(env.trainable_agents)

#     feedback = simultaneous(
#         vec_env,
#         num_episodes=2,
#         agent_interfaces=agent_interfaces,
#         fragment_length=rollout_configs["fragment_length"],
#         behavior_policies=behavior_policies,
#         agent_episodes=agent_episodes,
#         metric=metric,
#     )

#     logging.debug(f"feedback from simultaneous rollout: {feedback}")


# def test_sequential_rollout(poker_config):
#     creator = poker_aec_env.env
#     env = creator(**poker_config)

#     # generate agent interfaces
#     agent_interfaces = _gen_agent_interfaces(env)
#     _ = [interface.reset() for interface in agent_interfaces.values()]

#     # sample behavior policies
#     behavior_policies = {
#         agent: list(interface.policies.keys())[0]
#         for agent, interface in agent_interfaces.items()
#     }

#     # rollout configs
#     fragment_length = 100
#     num_episodes = 2
#     metric = get_metric("simple")(env.trainable_agents)

#     # generate agent episodes
#     agent_episodes = _gen_agent_episodes(
#         env.trainable_agents,
#         behavior_policies,
#         "poker",
#         fragment_length * num_episodes,
#         env.extra_returns,
#         is_sequential=True,
#     )

#     # sequential rollout
#     feedback = sequential(
#         env._env,
#         num_episodes=num_episodes,
#         agent_interfaces=agent_interfaces,
#         fragment_length=fragment_length,
#         behavior_policies=behavior_policies,
#         agent_episodes=agent_episodes,
#         metric=metric,
#     )

#     logging.debug(f"feedback from sequential rollout: {feedback}")
#     _ = [episode.clean_data() for episode in agent_episodes.values()]
