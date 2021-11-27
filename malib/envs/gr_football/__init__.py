import numpy as np

from malib.utils.preprocessor import get_preprocessor
from malib.utils.episode import EpisodeKey
from .env import BaseGFootBall, ParameterSharingWrapper


def env(**kwargs):
    return ParameterSharingWrapper(BaseGFootBall(**kwargs), lambda x: x[:6])


def default_config_gen():
    default_config = {
        # env building config
        "env_id": "Gfootball",
        "use_built_in_GK": True,
        "scenario_config": {
            "env_name": "5_vs_5",
            "number_of_left_players_agent_controls": 4,
            "number_of_right_players_agent_controls": 4,
            "representation": "raw",
            "logdir": "",
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "stacked": False,
        },
    }
    return default_config


def env_desc_gen(config):
    default_config = default_config_gen()
    default_config.update(config)
    env_for_spec = env(**config)
    env_desc = {
        "creator": env,
        "possible_agents": env_for_spec.possible_agents,
        "action_spaces": env_for_spec.action_spaces,
        "observation_spaces": env_for_spec.observation_spaces,
        "state_spaces": env_for_spec.state_space,
        "config": default_config,
    }
    env_for_spec.close()
    return env_desc


def build_sampler_config(
    env_desc,
    actor_rnn_state_shape,
    critic_rnn_state_shape,
):
    env_for_spec = env(**env_desc["config"])
    num_agents_share = env_for_spec.num_agent_share
    env_for_spec.close()
    observation_spaces = env_desc["observation_spaces"]
    env_desc["data_shapes"] = {}
    for aid, obsp in observation_spaces.items():
        num_ps = num_agents_share[aid]
        acsp, stsp = env_desc["action_spaces"][aid], env_desc["state_spaces"][aid]
        stsp_prep = get_preprocessor(stsp)(stsp)
        sampler_config = {
            "dtypes": {
                EpisodeKey.REWARD: np.float,
                EpisodeKey.DONE: np.bool,
                EpisodeKey.CUR_OBS: np.float,
                EpisodeKey.ACTION: np.int,
                EpisodeKey.ACTION_DIST: np.float,
                "active_mask": np.float,
                "available_action": np.float,
                "value": np.float,
                "return": np.float,
                "share_obs": np.float,
                "actor_rnn_states": np.float,
                "critic_rnn_states": np.float,
            },
            "data_shapes": {
                EpisodeKey.REWARD: (1,),
                EpisodeKey.DONE: (1,),
                EpisodeKey.CUR_OBS: obsp.shape,
                EpisodeKey.ACTION: (1,),
                EpisodeKey.ACTION_DIST: (acsp.n,),
                "active_mask": (1,),
                "available_action": (acsp.n,),
                "value": (1,),
                "return": (1,),
                "share_obs": stsp_prep.shape,
                "actor_rnn_states": (1, actor_rnn_state_shape),
                "critic_rnn_states": (1, critic_rnn_state_shape),
            },
        }
        for k in sampler_config["data_shapes"]:
            sampler_config["data_shapes"][k] = (num_ps,) + sampler_config[
                "data_shapes"
            ][k]
        env_desc["data_shapes"][aid] = sampler_config["data_shapes"]
