# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
import numpy as np

from gym import spaces
from numpy.core.fromnumeric import mean

from malib.utils.typing import AgentID, Dict, Any
from malib.utils.episode import Episode
from malib.rollout.envs.env import Environment
from malib.rollout.envs.gr_football.encoders import encoder_basic, rewarder_basic
from malib.rollout.envs.gr_football.encoders.state import State

try:
    from gfootball import env as raw_grf_env
except ImportError as e:
    raise e(
        "Please install Google football evironment before use: https://github.com/google-research/football"
    ) from None


class BaseGFootBall(Environment):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_id: str, use_built_in_GK: bool = True, scenario_configs={}):
        super(BaseGFootBall, self).__init__(
            env_id=env_id,
            use_built_in_GK=use_built_in_GK,
            scenario_configs=scenario_configs,
        )
        self._env_id = env_id
        self._use_built_in_GK = use_built_in_GK
        print("cfg =", scenario_configs)
        self._raw_env = raw_grf_env.create_environment(**scenario_configs)
        self.kwarg = scenario_configs

        self._num_left = self.kwarg["number_of_left_players_agent_controls"]
        self._num_right = self.kwarg["number_of_right_players_agent_controls"]
        self._include_GK = self._num_right == 5 or self._num_left == 5
        self._use_built_in_GK = (not self._include_GK) or self._use_built_in_GK

        if self._include_GK and self._use_built_in_GK:
            assert (
                scenario_configs["env_name"] == "5_vs_5"
            ), "currently only support a very specific scenario"
            assert self._num_left == 5 or self._num_left == 0
            assert self._num_right == 5 or self._num_right == 0
            self._num_right = self._num_right - 1
            self._num_left = self._num_left - 1

        self.possible_players = {
            "team_0": [
                f"team_0_player_{i + int(self._use_built_in_GK)}"
                for i in range(self._num_left)
            ],
            "team_1": [
                f"team_1_player_{i + int(self._use_built_in_GK)}"
                for i in range(self._num_right)
            ],
        }
        self.possible_teams = ["team_0", "team_1"]

        self._repr_mode = scenario_configs["representation"]

        if (
            "other_config_options" in scenario_configs
            and "reward_shaping_config" in scenario_configs["other_config_options"]
        ):
            RS_config = scenario_configs["other_config_options"][
                "reward_shaping_config"
            ]
        else:
            RS_config = None
        self._rewarder = rewarder_basic.Rewarder(RS_config)

        # jh: this is default, but we won't use this.
        self._feature_encoder = encoder_basic.FeatureEncoder()
        self._left_feature_encoder = self._feature_encoder
        self._right_feature_encoder = self._feature_encoder

        self.n_agents = len(self.possible_agents)

        self._build_interacting_spaces()
        self.max_step = 3001

        # if 'other_config_options' in scenario_configs and 'share_obs' in scenario_configs[
        #     'other_config_options']:
        #     self.if_share_obs = scenario_configs['other_config_options']["share_obs"]
        # else:
        #     self.if_share_obs = True

        # TODO(jh): try share obs again later
        # assert not self.if_share_obs, "jh: share_obs is not supported now. if need, remeber to modify policy.py."

    def record_episode_info_step(self, state, obs, rewards, dones, info):
        # super(BaseGFootBall, self).record_episode_info_step(rets)

        reward_ph = self.episode_metrics["agent_reward"]
        step_ph = self.episode_metrics["agent_step"]
        for aid, r in rewards.items():
            if aid not in reward_ph:
                reward_ph[aid] = []
                step_ph[aid] = 0
            reward_ph[aid].append(r)
            step_ph[aid] += 1

        self.episode_meta_info["env_done"] = dones["__all__"]
        self.episode_metrics["env_step"] += 1
        self.episode_metrics["episode_reward"] += sum(rewards.values())

        info = list(info.values())[0]
        self.custom_metrics["win"] += info.get("win", 0.0)
        self.custom_metrics["score"] += info.get("score", 0.0)
        self.custom_metrics["goal_diff"] += info.get("goal_diff", 0.0)
        self.custom_metrics["my_goal"] += info.get("my_goal", 0.0)
        self.custom_metrics["lose"] += info.get("lose", 0.0)

    def seed(self, seed=None):
        self._raw_env.seed(seed)
        return self.reset()

    @property
    def possible_agents(self):
        res = []
        for team_players in self.possible_players.values():
            res.extend(team_players)
        return res

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    def time_step(self, action_dict: Dict[AgentID, Any]):
        action_list = []
        if self._include_GK and self._use_built_in_GK and self._num_left > 0:
            action_list.append(19)
        for i, player_id in enumerate(sorted(action_dict)):
            if self._include_GK and self._use_built_in_GK and i == self._num_left:
                # which means action_dict is of size greater than num_left,
                #  and the first one is the goal keeper
                action_list.append(19)
            action_list.append(action_dict[player_id])

        obs, rew, done, info = self._raw_env.step(action_list)

        self.step_cnt += 1
        if self.step_cnt == 100:
            print(1)

        # jh: we don't need this deepcopy
        # obs = copy.deepcopy(obs)  # since the underlying env have cache obs
        rew = rew.tolist()
        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(obs)
            self._pop_list_for_built_in_GK(rew)

        reward = [
            self._rewarder.calc_reward(_r, _prev_obs, _obs, _action, _id)
            for _r, _prev_obs, _obs, _action, _id in zip(
                rew, self._prev_obs, obs, action_list, action_dict.keys()
            )
        ]

        assert len(obs) == len(action_list) and len(action_list) == len(self.states)
        for o, a, s in zip(obs, action_list, self.states):
            s.update_action(a)
            s.update_obs(o)

        self._prev_obs = obs

        reward = self._wrap_list_to_dict(reward)
        done = self._wrap_list_to_dict([done] * len(obs))

        info = [info.copy() for _ in range(len(obs))]
        env_done = self.env_done_check(done)

        score = []  # need to be list as it will need to collect state from both team
        goal_diff = []
        my_goal = []
        win = []
        lose = []
        if env_done:
            for i, _obs in enumerate(obs):
                my_score, opponent_score = _obs["score"]
                if my_score > opponent_score:
                    score.append(1.0)
                    win.append(1.0)
                    lose.append(0.0)
                    # win = 1.0
                elif my_score == opponent_score:
                    score.append(0.5)
                    win.append(0.0)
                    lose.append(0.0)
                    # win = 0.
                else:
                    score.append(0.0)
                    win.append(0.0)
                    lose.append(1.0)
                    # win = 0.
                # goal_diff = my_score - opponent_score
                goal_diff.append(my_score - opponent_score)
                my_goal.append(my_score)
                # info[i]["score"] = score
                # info[i]["win"] = int(score == 1.0)
                # info[i]["goal_diff"] = goal_diff

        else:
            score = [0.0] * len(obs)
            goal_diff = [0.0] * len(obs)
            win = [0.0] * len(obs)
            my_goal = [0.0] * len(obs)
            lose = [0.0] * len(obs)

        # steps_left = []
        # for i, _obs in enumerate(obs):
        #     steps_left.append([_obs["steps_left"]])
        steps_left = obs[0]["steps_left"]

        # reward = self._wrap_list_to_dict(reward)
        # done = self._wrap_list_to_dict([done] * len(obs))
        # info = [info.copy() for _ in range(len(obs))]
        for i, _info in enumerate(info):
            _info["score"] = score[i]
            _info["goal_diff"] = goal_diff[i]
            _info["steps_left"] = steps_left
            _info["win"] = win[i]
            _info["my_goal"] = my_goal[i]
            _info["lose"] = lose[i]

        info = self._wrap_list_to_dict(info)

        rets = {
            Episode.NEXT_OBS: self._get_obs(),
            Episode.REWARD: reward,
            Episode.DONE: done,
            Episode.INFO: info,
        }
        # If use the default feature encoder, the first num_action dimension is actually available action.
        if self._repr_mode == "raw" and isinstance(
            self._feature_encoder, encoder_basic.FeatureEncoder
        ):
            rets[Episode.ACTION_MASK] = {
                k: v[: self._num_actions] for k, v in rets[Episode.NEXT_OBS].items()
            }
        # if env_done:
        #     print('rets goal diff=', rets['infos']['team_0_player_1']['goal_diff'])
        # return rets                                                           #(state, observation, reward, done, info)
        return rets[Episode.NEXT_OBS], rets[Episode.NEXT_OBS], reward, done, info

    def close(self):
        self._raw_env.close()

    def reset(self, max_step: int = None):
        super(BaseGFootBall, self).reset(max_step=max_step)
        self.step_cnt = 0

        self.custom_metrics.update(
            {
                "total_reward": 0.0,
                "win": 0.0,
                "score": 0.0,
                "goal_diff": 0.0,
                "my_goal": 0.0,
                "lose": 0.0,
            }
        )

        obs = self._raw_env.reset()

        self.states = [State() for i in range(self.n_agents)]
        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(obs)

        assert len(obs) == len(self.states)
        for o, s in zip(obs, self.states):
            s.update_obs(o)

        self._prev_obs = obs

        self.agents = self.possible_agents
        self.dones = dict(zip(self.agents, [False] * self.n_agents))
        self.scores = dict(zip(self.agents, [{"scores": [0.0]}] * self.n_agents))
        rets = {Episode.CUR_OBS: self._get_obs()}
        if self._repr_mode == "raw" and isinstance(
            self._feature_encoder, encoder_basic.FeatureEncoder
        ):
            rets[Episode.ACTION_MASK] = {
                k: v[: self._num_actions] for k, v in rets[Episode.CUR_OBS].items()
            }
        return rets[Episode.ACTION_MASK], rets[Episode.CUR_OBS]

    def _build_interacting_spaces(self):
        possible_agents = self.possible_agents
        # NOTE(jh): we can use action 20 if set action_set to v2
        # this just restrict model to predict from 19 actions by default
        self._num_actions = 19  # self._raw_env.env._num_actions

        assert self._repr_mode == "raw", "we only support raw format observation"

        self._action_spaces = {
            player_id: spaces.Discrete(self._num_actions)
            for player_id in possible_agents
        }

        if self._repr_mode == "raw":
            self._feature_encoder = encoder_basic.FeatureEncoder()
        obs = self._raw_env.reset()

        self.states = [State() for i in range(self.n_agents)]

        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(obs)
        assert len(obs) == len(self.states)
        for o, s in zip(obs, self.states):
            s.update_obs(o)

        obs = self._get_obs()

        self._observation_spaces = {
            player_id: spaces.Box(
                low=-10.0, high=10.0, shape=obs[player_id].shape, dtype=np.float32
            )
            for player_id in self.possible_agents
        }

    def _get_obs(self):
        if self._repr_mode == "raw":
            obs = self._build_observation_from_raw()
        else:
            assert not self._use_built_in_GK
            obs = self._raw_env.observation()
        return self._wrap_list_to_dict(obs)

    def _build_observation_from_raw(self):
        """
        get the observation of all player's in teams
        """

        assert (
            self._num_actions == 19
        ), "jh: sorry (!-_-)>, 20 is not supported now! because of my modifications to codes..."

        states = self.states.copy()

        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(states)

        left_states = states[: self._num_left]
        if len(left_states) != 0:
            left_encoded_obs_list = self._left_feature_encoder.encode(left_states)
        else:
            left_encoded_obs_list = []

        right_states = states[self._num_left :]
        if len(right_states) != 0:
            right_encoded_obs_list = self._right_feature_encoder.encode(right_states)
        else:
            right_encoded_obs_list = []

        encode_obs_list = left_encoded_obs_list + right_encoded_obs_list

        return encode_obs_list

    def _wrap_list_to_dict(self, data_in_list):
        return dict(zip(self.possible_agents, data_in_list))

    def _pop_list_for_built_in_GK(self, data_list):
        assert self._include_GK and self._use_built_in_GK
        if self._num_left > 0:
            data_list.pop(0)
        if self._num_right > 0:
            data_list.pop(-self._num_right)


# def ParameterSharing(base_env: BaseGFootBall, parameter_sharing_mapping):
#     class Env(Environment):
#         def __init__(self):
#             """
#             :param base_env: the environment where agents share their parameters
#             :param parameter_sharing_mapping: how to share the parameters
#             """
#             super().__init__(**{})
#             self._env = base_env
#
#             self._ps_mapping_func = parameter_sharing_mapping
#             self._ps_buckets = {aid: [] for aid in self.possible_agents}
#             for aid in base_env.possible_agents:
#                 self._ps_buckets[parameter_sharing_mapping(aid)].append(aid)
#             self._ps_buckets = {
#                 aid: sorted(self._ps_buckets[aid]) for aid in self.possible_agents
#             }
#
#             self.num_agent_share = {aid: len(v) for aid, v in self._ps_buckets.items()}
#
#             self.is_sequential = False
#             self.max_step = self._env.max_step
#
#             self.if_share_obs = self._env.if_share_obs
#
#         @property
#         def observation_spaces(self) -> Dict[AgentID, gym.Space]:
#             return {
#                 aid: base_env.observation_spaces[self._ps_buckets[aid][0]]
#                 for aid in self.possible_agents
#             }
#
#         @property
#         def action_spaces(self) -> Dict[AgentID, gym.Space]:
#             return {
#                 aid: base_env.action_spaces[self._ps_buckets[aid][0]]
#                 for aid in self.possible_agents
#             }
#
#         @property
#         def possible_agents(self) -> List[AgentID]:
#             return sorted(
#                 list(
#                     set(
#                         parameter_sharing_mapping(aid)
#                         for aid in base_env.possible_agents
#                     )
#                 )
#             )
#
#         @property
#         def state_spaces(self):
#             concate_stsp = {
#                 aid: gym.spaces.Dict(
#                     {
#                         player_id: obssp
#                         for player_id, obssp in self._env.observation_spaces.items()
#                         if aid in player_id
#                     }
#                 )
#                 for aid in self.possible_agents
#             }
#
#             if not self.if_share_obs:
#                 return concate_stsp
#             else:
#                 return {
#                     k: get_preprocessor(v)(v).observation_space
#                     for k, v in concate_stsp.items()
#                 }
#
#         def _build_state_from_obs(self, obs):
#             if not self.if_share_obs:
#                 return obs  # if not sharing the observation
#
#             state = {
#                 aid: np.reshape(obs[aid], [1, np.prod(obs[aid].shape)])
#                 for aid in self.possible_agents
#             }
#             for aid, share_obs in state.items():
#                 state[aid] = np.repeat(share_obs, obs[aid].shape[0], axis=0)
#             return state
#
#         def action_adapter(
#                 self, policy_outputs: Dict[str, Dict[AgentID, Any]], **kwargs
#         ):
#             """Convert policy action to environment actions. Default by policy action"""
#
#             return policy_outputs["action"]
#
#         def reset(
#                 self, max_step: int = None
#         ):
#             super().reset(max_step)
#
#             self.custom_metrics.update(
#                 {
#                     aid: {
#                         "total_reward": 0.0,
#                         "win": 0.0,
#                         "score": 0.0,
#                         "goal_diff": 0.0,
#                         "my_goal": 0.0,
#                         'lose': 0.0,
#                     }
#                     for aid in self.possible_agents
#                 }
#             )
#
#             rets = self._env.reset(max_step)
#
#             def f(x_dict, reduce_func):
#                 x_dict = self._build_from_base_dict(x_dict)
#                 return {k: reduce_func(v) for k, v in x_dict.items()}
#
#             rets = {k: f(v, np.vstack) for k, v in rets.items()}
#             # rets[EpisodeKey.CUR_STATE] = self._build_state_from_obs(
#             #     rets[EpisodeKey.CUR_OBS]
#             # )
#
#             if Episode.ACTION_MASK in rets:
#                 rets[Episode.ACTION_MASK] = f(
#                     rets[Episode.ACTION_MASK], np.vstack
#                 )
#
#             return rets
#
#         def seed(self, seed=None):
#             self._env.seed(seed)
#
#         def _build_from_base_dict(self, base_dict: Dict[AgentID, Any]):
#             """Map the entry of a dict to a list with defined mapping function."""
#             ans = {aid: [] for aid in self.possible_agents}
#             for base_aid, data in base_dict.items():
#                 if base_aid == "__all__":  # (ziyu): This is for done['__all__']
#                     continue
#                 ans[self._ps_mapping_func(base_aid)].append(data)
#
#             return ans
#
#         def time_step(self, action_dict):
#             base_action = self._extract_to_base(action_dict)
#             rets = self._env.step(base_action)
#
#             # print('wrapper rets goal diff = ', rets['infos']['team_0_player_1']['goal_diff'])
#             # print('raw rets = ', rets)
#             # raise NotImplementedError
#
#             def f(x_dict, reduce_func):
#                 x_dict = self._build_from_base_dict(x_dict)
#                 return {k: reduce_func(v) for k, v in x_dict.items()}
#
#             (obs, reward, done, info) = (
#                 f(rets[Episode.NEXT_OBS], np.vstack),
#                 f(rets[Episode.REWARD], np.vstack),
#                 f(rets[Episode.DONE], np.vstack),
#                 f(rets[Episode.INFO], lambda x: x[0]),
#             )
#             done["__all__"] = rets[Episode.DONE]["__all__"]
#             # if np.all(done['team_0']):
#             # print('wrapper rets goal diff = ', rets['infos'])
#             # print('wrapper rets reward = ', reward)
#             # print('done = ', np.all(done['team_0']))
#
#             for aid, action in action_dict.items():
#                 info[aid]["num_pass"] = np.logical_and(action <= 11, action >= 9).sum()
#                 info[aid]["num_shot"] = (action == 12).sum()
#
#             res = {
#                 Episode.NEXT_OBS: obs,
#                 # EpisodeKey.NEXT_STATE: self._build_state_from_obs(obs),
#                 Episode.REWARD: reward,
#                 Episode.DONE: done,
#                 Episode.INFO: info,
#             }
#
#             if Episode.ACTION_MASK in rets:
#                 res[Episode.ACTION_MASK] = f(rets[Episode.ACTION_MASK], np.vstack)
#
#             return res
#
#         def env_done_check(self, agent_dones: Dict[AgentID, np.ndarray]) -> bool:
#             return agent_dones["__all__"]
#
#         def _extract_to_base(self, from_dict):
#             to_dict = {}
#             for aid, bucket in self._ps_buckets.items():
#                 for i, base_aid in enumerate(bucket):
#                     to_dict[base_aid] = from_dict[aid][i]
#             return to_dict
#
#         def close(self):
#             self._env.close()
#
#         def record_episode_info_step(self, rets):
#             super().record_episode_info_step(rets)
#             # if self.custom_metrics['team_0']['goal_diff']!= 0:
#             #    print('##################################################custom metric = ', self.custom_metrics['team_0']['goal_diff'])
#             # raise NotImplementedError
#
#             reward = rets[Episode.REWARD]
#             info = rets[Episode.INFO]
#
#             for aid in reward:
#                 self.custom_metrics[aid]["total_reward"] += reward[aid].sum()
#                 self.custom_metrics[aid]["win"] += info[aid].get("win", 0.0)
#                 self.custom_metrics[aid]["score"] += info[aid].get("score", 0.0)
#                 self.custom_metrics[aid]["goal_diff"] += info[aid].get("goal_diff",
#                                                                        0.0)  # here determined the tensorboard log
#                 # self.custom_metrics[aid]["goal_diff"] = 100
#                 self.custom_metrics[aid]["num_pass"] += info[aid]["num_pass"]
#                 self.custom_metrics[aid]["num_shot"] += info[aid]["num_shot"]
#                 self.custom_metrics[aid]['my_goal'] += info[aid].get('my_goal', 0.0)
#                 self.custom_metrics[aid]["lose"] += info[aid].get("lose", 0.0)
#
#
#     return Env()


if __name__ == "__main__":  # fixme(yan): ParameterSharingWrapper(env) not working

    default_config = {
        # env building config
        "env_id": "Gfootball",
        "use_built_in_GK": True,
        "scenario_configs": {
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
    # from malib.envs.gr_football.vec_wrapper import DummyVecEnv

    # default_config["other_config_options"] = {"action_set": "v2"}
    default_config["scenario_configs"]["env_name"] = "5_vs_5"
    default_config["scenario_configs"]["number_of_left_players_agent_controls"] = 4
    default_config["scenario_configs"]["number_of_right_players_agent_controls"] = 0
    default_config["use_built_in_GK"] = True
    default_config["scenario_configs"]["render"] = True
    default_config["scenario_configs"]["other_config_options"] = {
        "action_set": "v2"
    }  # ['action_set'] = "v2"
    default_config["scenario_configs"]["other_config_options"]["share_obs"] = False

    # default_config['other_config_options'] = {}

    def env_fn():
        env = BaseGFootBall(**default_config)
        # env = ParameterSharingWrapper(env) #, lambda x: x[:6])
        return env

    # env = DummyVecEnv([env_fn] * 2)
    env = env_fn()
    print(env.possible_agents)
    print(env.observation_spaces)
    print(env.action_spaces)
    env.reset()
    done = False
    while not done:
        actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
        out = env.step(actions)
        # done = out['done']['team_0_player_1']
        # if out['done']['team_0_player_1']:
        #     print('terminal info =', out['infos']['team_0_player_1']['goal_diff'])
