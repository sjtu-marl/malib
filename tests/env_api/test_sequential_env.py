import pytest
import importlib
import numpy as np

from malib.utils.typing import Dict
from malib.utils.episode import EpisodeKey

from .utils import simple_discrete_agent


@pytest.mark.parametrize(
    "module_path,cname,env_id,scenario_configs",
    [
        ("malib.envs.poker", "PokerParallelEnv", "leduc_poker", {"fixed_player": True}),
    ],
    scope="class",
)
def test_env(module_path, cname, env_id, scenario_configs):
    creator = getattr(importlib.import_module(module_path), cname)
    env = creator(env_id=env_id, scenario_configs=scenario_configs)

    assert env.is_sequential

    possible_agents = env.possible_agents
    obs_spaces = env.observation_spaces
    action_spaces = env.action_spaces

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

    acts = {env.agent_selection: action_spaces[env.agent_selection].sample()}

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

    obs = rets[EpisodeKey.CUR_OBS]
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
        ("malib.envs.poker", "PokerParallelEnv", "leduc_poker", {"fixed_player": True}),
    ],
)
def test_rollout(module_path, cname, env_id, scenario_configs):
    creator = getattr(importlib.import_module(module_path), cname)
    env = creator(env_id=env_id, scenario_configs=scenario_configs)
    assert env.is_sequential

    rets = env.reset()
    agents = {
        aid: simple_discrete_agent(
            aid, env.observation_spaces[aid], env.action_spaces[aid]
        )
        for aid in env.possible_agents
    }
    for _ in range(10):
        acts = {env.agent_selection: agents[env.agent_selection](rets)}
        rets = env.step(acts)
        done = rets[EpisodeKey.DONE]
        done = any(done.values())
        if done:
            break
    env.close()
