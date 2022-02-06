import pytest
import importlib
import numpy as np
import gym.spaces
from malib.utils.typing import Dict
from malib.utils.episode import EpisodeKey

from .utils import simple_discrete_agent


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}),
        ("malib.envs.mpe", "MPE", "simple_spread_v2", {"max_cycles": 25}),
        (
            "malib.envs.gr_football",
            "BaseGFootBall",
            "Gfootball",
            {
                "env_name": "academy_run_pass_and_shoot_with_keeper",
                "number_of_left_players_agent_controls": 2,
                "number_of_right_players_agent_controls": 1,
                "representation": "raw",
                "logdir": "",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
                "stacked": False,
            },
        ),
        (
            "malib.envs.maatari",
            "MAAtari",
            "basketball_pong_v2",
            {
                "wrappers": [
                    {"name": "resize_v0", "params": [84, 84]},
                    {"name": "dtype_v0", "params": ["float32"]},
                    {
                        "name": "normalize_obs_v0",
                        "params": {"env_min": 0.0, "env_max": 1.0},
                    },
                ],
                "obs_type": "grayscale_image",
                "num_players": 2,
            },
        ),
        ("malib.envs.star_craft2", "SC2Env", "3m", {"map_name": "3m"}),
    ],
    scope="class",
)
def test_env(module_path, cname, env_id, scenario_configs):
    creator = getattr(importlib.import_module(module_path), cname)
    env = creator(env_id=env_id, scenario_configs=scenario_configs)

    assert not env.is_sequential

    possible_agents = env.possible_agents
    obs_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    agents = {
        aid: simple_discrete_agent(
            aid, env.observation_spaces[aid], env.action_spaces[aid]
        )
        for aid in env.possible_agents
    }

    rets = env.reset()
    assert isinstance(rets, Dict) and EpisodeKey.CUR_OBS in rets, rets
    agent_obs = rets[EpisodeKey.CUR_OBS]

    for aid, obs in agent_obs.items():
        assert (
            aid in possible_agents
        ), "Illegal agent key: {}, expected keys: {}".format(aid, possible_agents)
        assert obs_spaces[aid].contains(
            obs
        ), "Reset observation: {!r} of agent: {!r} not in space".format(obs, aid)

    acts = {aid: agent(rets) for aid, agent in agents.items()}

    rets = env.step(acts)

    rets[EpisodeKey.DONE].pop("__all__")

    aids_check_buffer = []
    for k, v in rets.items():
        aids_check_buffer.append((k, sorted(list(v.keys()))))
    ks, vs = list(zip(*aids_check_buffer))
    for i, _vs in enumerate(vs):
        assert (
            vs[0] == _vs
        ), "Inconsistency in the stepping returns when key={} expect: {} while vs={}".format(
            ks[i], vs[0], _vs
        )

    rewards = rets[EpisodeKey.REWARD]
    assert isinstance(rewards, Dict), rewards
    for aid, r in rewards.items():
        assert (
            aid in possible_agents
        ), "Illegal agent key: {}, expected keys: {}".format(aid, possible_agents)
        assert np.isscalar(r), "agent's reward {} is not a scalar for {}".format(
            aid, r, env
        )

    obs = rets.get(EpisodeKey.NEXT_OBS)
    assert isinstance(obs, Dict), obs
    for aid, _obs in obs.items():
        assert (
            aid in possible_agents
        ), "Illegal agent key: {}, expected keys: {}".format(aid, possible_agents)
        assert obs_spaces[aid].contains(_obs), (aid, _obs)

    env.close()


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        ("malib.envs.gym", "GymEnv", "CartPole-v0", {}),
        ("malib.envs.mpe", "MPE", "simple_push_v2", {"max_cycles": 25}),
        ("malib.envs.mpe", "MPE", "simple_spread_v2", {"max_cycles": 25}),
        (
            "malib.envs.gr_football",
            "BaseGFootBall",
            "Gfootball",
            {
                "env_name": "academy_run_pass_and_shoot_with_keeper",
                "number_of_left_players_agent_controls": 2,
                "number_of_right_players_agent_controls": 1,
                "representation": "raw",
                "logdir": "",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
                "stacked": False,
            },
        ),
        (
            "malib.envs.maatari",
            "MAAtari",
            "basketball_pong_v2",
            {
                "wrappers": [
                    {"name": "resize_v0", "params": [84, 84]},
                    {"name": "dtype_v0", "params": ["float32"]},
                    {
                        "name": "normalize_obs_v0",
                        "params": {"env_min": 0.0, "env_max": 1.0},
                    },
                ],
                "obs_type": "grayscale_image",
                "num_players": 2,
            },
        ),
        ("malib.envs.star_craft2", "SC2Env", "3m", {"map_name": "3m"}),
    ],
)
def test_rollout(module_path, cname, env_id, scenario_configs):
    creator = getattr(importlib.import_module(module_path), cname)
    env = creator(env_id=env_id, scenario_configs=scenario_configs)

    assert not env.is_sequential

    rets = env.reset()
    agents = {
        aid: simple_discrete_agent(
            aid, env.observation_spaces[aid], env.action_spaces[aid]
        )
        for aid in env.possible_agents
    }
    for _ in range(10):
        acts = {aid: agent_step(rets) for aid, agent_step in agents.items()}
        rets = env.step(acts)
        done = rets[EpisodeKey.DONE]
        done = any(done.values())
        if done:
            break
    env.close()
