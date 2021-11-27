import copy
import gym
import numpy as np

from gym import spaces
from numpy.core.fromnumeric import mean


from malib.utils.typing import AgentID, Callable, Dict, Any, Union, List
from malib.utils.logger import Logger
from malib.utils.episode import Episode, EpisodeKey
from malib.envs.env import Environment
from malib.envs.gr_football.encoders import encoder_basic, rewarder_basic

try:
    from gfootball import env as raw_grf_env
except Exception:
    Logger.error(
        "Please install Google football evironment before use: https://github.com/google-research/football"
    )


class BaseGFootBall(Environment):
    metadata = {"render.modes": ["human"]}

    def record_episode_info_step(self, rets):
        super(BaseGFootBall, self).record_episode_info_step(rets)

        reward = rets[EpisodeKey.REWARD]
        info = list(rets[EpisodeKey.INFO].values())[0]
        self.custom_metrics["total_reward"] += mean(list(reward.values()))
        self.custom_metrics["win"] += info.get("win", 0.0)
        self.custom_metrics["score"] += info.get("score", 0.0)
        self.custom_metrics["goal_diff"] += info.get("goal_diff", 0.0)
        self.custom_metrics["num_pass"] += info.get("num_pass", 0.0)
        self.custom_metrics["num_shot"] += info.get("num_shot", 0.0)

    def __init__(self, env_id: str, use_built_in_GK: bool = True, scenario_configs={}):
        super(BaseGFootBall, self).__init__(
            env_id=env_id,
            use_built_in_GK=use_built_in_GK,
            scenario_configs=scenario_configs,
        )
        self._env_id = env_id
        self._use_built_in_GK = use_built_in_GK
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
                f"team_0_player_{i+int(self._use_built_in_GK)}"
                for i in range(self._num_left)
            ],
            "team_1": [
                f"team_1_player_{i+int(self._use_built_in_GK)}"
                for i in range(self._num_right)
            ],
        }
        self.possible_teams = ["team_0", "team_1"]

        self._repr_mode = scenario_configs["representation"]
        self._rewarder = rewarder_basic.Rewarder()

        self.n_agents = len(self.possible_agents)

        self._build_interacting_spaces()

    def seed(self, seed=None):
        self._raw_env.seed(seed)
        return self.reset()

    @property
    def possible_agents(self) -> List[AgentID]:
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

        obs = copy.deepcopy(obs)  # since the underlying env have cache obs
        rew = rew.tolist()
        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(obs)
            self._pop_list_for_built_in_GK(rew)

        reward = [
            self._rewarder.calc_reward(_r, _prev_obs, _obs)
            for _r, _prev_obs, _obs in zip(rew, self._prev_obs, obs)
        ]
        self._prev_obs = obs

        reward = self._wrap_list_to_dict(reward)
        done = self._wrap_list_to_dict([done] * len(obs))

        info = [info.copy() for _ in range(len(obs))]
        env_done = self.env_done_check(done)

        if env_done:
            for i, _obs in enumerate(obs):
                my_score, opponent_score = _obs["score"]
                if my_score > opponent_score:
                    score = 1.0
                elif my_score == opponent_score:
                    score = 0.5
                else:
                    score = 0.0
                goal_diff = my_score - opponent_score
                info[i]["score"] = score
                info[i]["win"] = int(score == 1.0)
                info[i]["goal_diff"] = goal_diff

        info = self._wrap_list_to_dict(info)

        return {
            EpisodeKey.NEXT_OBS: self._get_obs(),
            EpisodeKey.REWARD: reward,
            EpisodeKey.DONE: done,
            EpisodeKey.INFO: info,
        }

    def close(self):
        self._raw_env.close()

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super(BaseGFootBall, self).reset(
            max_step=max_step, custom_reset_config=custom_reset_config
        )

        self.custom_metrics.update(
            {
                "total_reward": 0.0,
                "win": 0.0,
                "score": 0.0,
                "goal_diff": 0.0,
                "num_pass": 0.0,
                "num_shot": 0.0,
            }
        )

        self._prev_obs = self._raw_env.reset()
        self.agents = self.possible_agents
        self.dones = dict(zip(self.agents, [False] * self.n_agents))
        self.scores = dict(zip(self.agents, [{"scores": [0.0]}] * self.n_agents))

        return {EpisodeKey.CUR_OBS: self._get_obs()}

    def _build_interacting_spaces(self):
        possible_agents = self.possible_agents
        self._num_actions = 19

        self._action_spaces = {
            player_id: spaces.Discrete(self._num_actions)
            for player_id in possible_agents
        }

        if self._repr_mode == "raw":
            self._feature_encoder = encoder_basic.FeatureEncoder()
        self._raw_env.reset()
        obs = self._get_obs()
        self._observation_spaces = {
            player_id: spaces.Box(
                low=-10.0, high=10.0, shape=obs[player_id].shape, dtype=np.float32
            )
            for player_id in possible_agents
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

        def encode_obs(raw_obs):
            obs = self._feature_encoder.encode(raw_obs)

            obs_cat = np.hstack(
                [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
            )

            return obs_cat

        raw_obs_list = self._raw_env.observation()

        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(raw_obs_list)
        encode_obs_list = [encode_obs(r_obs) for r_obs in raw_obs_list]

        return encode_obs_list

    def _wrap_list_to_dict(self, data_in_list):
        return dict(zip(self.possible_agents, data_in_list))

    def _pop_list_for_built_in_GK(self, data_list):
        assert self._include_GK and self._use_built_in_GK
        if self._num_left > 0:
            data_list.pop(0)
        if self._num_right > 0:
            data_list.pop(-self._num_right)


def ParameterSharingWrapper(
    base_env: BaseGFootBall, parameter_sharing_mapping: Callable
):
    class Env:
        def __init__(self):
            """
            :param base_env: the environment where agents share their parameters
            :param parameter_sharing_mapping: how to share the parameters
            """
            self._env = base_env
            self.possible_agents = sorted(
                list(
                    set(
                        parameter_sharing_mapping(aid)
                        for aid in base_env.possible_agents
                    )
                )
            )
            self._ps_mapping_func = parameter_sharing_mapping
            self._ps_buckets = {aid: [] for aid in self.possible_agents}
            for aid in base_env.possible_agents:
                self._ps_buckets[parameter_sharing_mapping(aid)].append(aid)
            self._ps_buckets = {
                aid: sorted(self._ps_buckets[aid]) for aid in self.possible_agents
            }
            self.action_spaces = {
                aid: base_env.action_spaces[self._ps_buckets[aid][0]]
                for aid in self.possible_agents
            }
            self.observation_spaces = {
                aid: base_env.observation_spaces[self._ps_buckets[aid][0]]
                for aid in self.possible_agents
            }
            self.num_agent_share = {aid: len(v) for aid, v in self._ps_buckets.items()}

            self.is_sequential = False

        @property
        def state_space(self):
            return {
                aid: gym.spaces.Dict(
                    {
                        player_id: obssp
                        for player_id, obssp in self._env.observation_spaces.items()
                        if aid in player_id
                    }
                )
                for aid in self.possible_agents
            }

        def _build_state_from_obs(self, obs):
            state = {
                aid: np.reshape(obs[aid], [1, np.prod(obs[aid].shape)])
                for aid in self.possible_agents
            }
            for aid, share_obs in state.items():
                state[aid] = np.repeat(share_obs, obs[aid].shape[0], axis=0)
            return state

        def record_episode_info(self, rew, info, act_dict):
            data_dict = copy.deepcopy(info)
            for aid, d in data_dict.items():
                act = act_dict[aid]
                d["num_shot"] = (act == 12).sum(-1)
                d["num_pass"] = np.logical_and(act <= 11, act >= 9).sum(axis=-1)
                self.episode_info[aid].step(rew[aid], d)

        def reset(self):
            self.episode_info = {aid: GRFEpisodeInfo() for aid in self.possible_agents}

            base_obs = self._env.reset()
            obs = self._build_from_base_dict(base_obs)
            obs = {aid: np.vstack(_obs) for aid, _obs in obs.items()}
            return {
                Episode.CUR_OBS: obs,
                Episode.CUR_STATE: self._build_state_from_obs(obs),
            }

        def seed(self, seed=None):
            base_obs = self._env.seed(seed)
            obs = self._build_from_base_dict(base_obs)
            obs = {aid: self.np.vstack(_obs) for aid, _obs in obs.items()}
            return {
                Episode.CUR_OBS: obs,
                Episode.CUR_STATE: self._build_state_from_obs(obs),
            }

        def _build_from_base_dict(self, base_dict):
            ans = {aid: [] for aid in self.possible_agents}
            for base_aid, data in base_dict.items():
                ans[self._ps_mapping_func(base_aid)].append(data)

            return ans

        def step(self, action_dict):
            base_action = self._extract_to_base(action_dict)
            obs, reward, done, info = self._env.step(base_action)

            def f(x_dict, reduce_func):
                x_dict = self._build_from_base_dict(x_dict)
                return {k: reduce_func(v) for k, v in x_dict.items()}

            (obs, reward, done, info) = (
                f(obs, np.vstack),
                f(reward, np.vstack),
                f(done, np.vstack),
                f(info, lambda x: x[0]),
            )

            self.record_episode_info(reward, info, action_dict)

            res = {
                Episode.CUR_OBS: obs,
                Episode.CUR_STATE: self._build_state_from_obs(obs),
                Episode.REWARD: reward,
                Episode.DONE: done,
                Episode.INFO: info,
            }
            return res

        def _extract_to_base(self, from_dict):
            to_dict = {}
            for aid, bucket in self._ps_buckets.items():
                for i, base_aid in enumerate(bucket):
                    to_dict[base_aid] = from_dict[aid][i]
            return to_dict

        def close(self):
            self._env.close()

    return Env()


if __name__ == "__main__":
    from malib.envs.gr_football import default_config
    from malib.envs.gr_football.vec_wrapper import DummyVecEnv

    # default_config["other_config_options"] = {"action_set": "v2"}
    default_config["scenario_config"][
        "env_name"
    ] = "academy_run_pass_and_shoot_with_keeper"
    default_config["scenario_config"]["number_of_left_players_agent_controls"] = 2
    default_config["scenario_config"]["number_of_right_players_agent_controls"] = 1
    default_config["use_built_in_GK"] = True

    def env_fn():
        env = BaseGFootBall(**default_config)
        env = ParameterSharingWrapper(env, lambda x: x[:6])
        return env

    env = DummyVecEnv([env_fn] * 2)
    print(env.possible_agents)
    print(env.observation_spaces)
    print(env.action_spaces)
    env.reset()
    # done = False
    # while True:
    #     actions = {aid: np.zeros(4, dtype=int) for aid in env.possible_agents}
    #     obs, reward, done, info = env.step(actions)
