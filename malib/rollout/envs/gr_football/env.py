from typing import Tuple, Callable, Dict, Any, Union, List

import copy
import gym
import numpy as np

from gym import spaces
from numpy.core.fromnumeric import mean

from malib.utils.typing import AgentID
from malib.utils.episode import Episode
from malib.utils.preprocessor import get_preprocessor
from malib.rollout.envs.env import Environment
from malib.rollout.envs.gr_football.encoders import encoder_basic, rewarder_basic

try:
    from gfootball import env as raw_grf_env
except ImportError as e:
    raise e(
        "Please install Google football evironment before use: https://github.com/google-research/football"
    ) from None


class GRFootball(Environment):
    metadata = {"render.modes": ["human"]}

    def record_episode_info_step(self, observations, rewards, dones, infos):
        super(GRFootball, self).record_episode_info_step(
            observations, rewards, dones, infos
        )

        reward = rewards
        info = list(infos.values())[0]

        self.custom_metrics["total_reward"] += mean(list(reward.values()))
        self.custom_metrics["win"] += info.get("win", 0.0)
        self.custom_metrics["score"] += info.get("score", 0.0)
        self.custom_metrics["goal_diff"] += info.get("goal_diff", 0.0)
        self.custom_metrics["num_pass"] += info.get("num_pass", 0.0)
        self.custom_metrics["num_shot"] += info.get("num_shot", 0.0)

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        self._observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        self._action_spaces

    def __init__(self, env_id: str, use_built_in_GK: bool = True, scenario_configs={}):
        super(GRFootball, self).__init__(
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
            assert (
                scenario_configs.get("other_config_options") is not None
            ), "You should specify `other_config_options`"
            assert (
                scenario_configs["other_config_options"].get("action_set") == "v2"
            ), "You should specify `action_set` in `other_config_options` with `v2`"

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
        if self._repr_mode == "raw":
            self._feature_encoder = encoder_basic.FeatureEncoder(
                self._num_left, self._num_right, n_action=19
            )
        else:
            raise ValueError(f"reprensentation mode: {self._repr_mode} is illegal.")

        self._num_actions = 19
        self._raw_env.reset()
        obs_dict = self._get_obs()
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    k: spaces.Box(low=-10.0, high=10.0, shape=v.shape)
                    for k, v in obs.items()
                }
            )
            for agent, obs in obs_dict.items()
        }
        self._action_spaces = {
            player_id: spaces.Discrete(self._num_actions)
            for player_id in self.possible_agents
        }
        self._rewarder = rewarder_basic.Rewarder()

        self.n_agents = len(self.possible_agents)

        self.max_step = 3000

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

    def time_step(
        self, action_dict: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
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
        return self._get_obs(), reward, done, info

    def close(self):
        self._raw_env.close()

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super(GRFootball, self).reset(
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

        observations = self._get_obs()
        action_masks = self.action_mask_extract(observations)

        return (observations, action_masks)

    def _get_obs(self):
        if self._repr_mode == "raw":
            obs = self._build_observation_from_raw()
        else:
            raise RuntimeError
        return self._wrap_list_to_dict(obs)

    def _build_observation_from_raw(self):
        """
        get the observation of all player's in teams
        """

        raw_obs_list = self._raw_env.observation()

        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(raw_obs_list)

        encode_obs_list = [self._feature_encoder.encode(x) for x in raw_obs_list]
        return encode_obs_list

    def _wrap_list_to_dict(self, data_in_list):
        return dict(zip(self.possible_agents, data_in_list))

    def _pop_list_for_built_in_GK(self, data_list):
        assert self._include_GK and self._use_built_in_GK
        if self._num_left > 0:
            data_list.pop(0)
        if self._num_right > 0:
            data_list.pop(-self._num_right)


class ParameterSharingEnv(Environment):
    def __init__(self, base_env, parameter_sharing_mapping):
        """
        :param base_env: the environment where agents share their parameters
        :param parameter_sharing_mapping: how to share the parameters
        """
        super().__init__(**{})
        self._env = base_env

        self._ps_mapping_func = parameter_sharing_mapping
        self._ps_buckets = {aid: [] for aid in self.possible_agents}
        for aid in base_env.possible_agents:
            self._ps_buckets[parameter_sharing_mapping(aid)].append(aid)
        self._ps_buckets = {
            aid: sorted(self._ps_buckets[aid]) for aid in self.possible_agents
        }

        self.num_agent_share = {aid: len(v) for aid, v in self._ps_buckets.items()}

        self.is_sequential = False
        self.max_step = self._env.max_step

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return {
            aid: self._env.observation_spaces[self._ps_buckets[aid][0]]
            for aid in self.possible_agents
        }

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return {
            aid: self._env.action_spaces[self._ps_buckets[aid][0]]
            for aid in self.possible_agents
        }

    @property
    def possible_agents(self) -> List[AgentID]:
        return sorted(
            list(set(self._ps_mapping_func(aid) for aid in self._env.possible_agents))
        )

    @property
    def state_spaces(self):
        concate_stsp = {
            aid: gym.spaces.Dict(
                {
                    player_id: obssp
                    for player_id, obssp in self._env.observation_spaces.items()
                    if aid in player_id
                }
            )
            for aid in self.possible_agents
        }
        return {
            k: get_preprocessor(v)(v).observation_space for k, v in concate_stsp.items()
        }

    def _build_state_from_obs(self, obs):
        state = {
            aid: np.reshape(obs[aid], [1, np.prod(obs[aid].shape)])
            for aid in self.possible_agents
        }
        for aid, share_obs in state.items():
            state[aid] = np.repeat(share_obs, obs[aid].shape[0], axis=0)
        return state

    def action_adapter(self, policy_outputs: Dict[str, Dict[AgentID, Any]], **kwargs):
        """Convert policy action to environment actions. Default by policy action"""

        return policy_outputs["action"]

    def reset(self, max_step: int = None, custom_reset_config: Dict[str, Any] = None):
        super().reset(max_step, custom_reset_config)

        self.custom_metrics.update(
            {
                aid: {
                    "total_reward": 0.0,
                    "win": 0.0,
                    "score": 0.0,
                    "goal_diff": 0.0,
                    "num_pass": 0.0,
                    "num_shot": 0.0,
                }
                for aid in self.possible_agents
            }
        )

        rets = self._env.reset(max_step, custom_reset_config)

        def f(x_dict, reduce_func):
            x_dict = self._build_from_base_dict(x_dict)
            return {k: reduce_func(v) for k, v in x_dict.items()}

        rets = {k: f(v, np.vstack) for k, v in rets.items()}
        rets[Episode.CUR_STATE] = self._build_state_from_obs(rets[Episode.CUR_OBS])

        if Episode.ACTION_MASK in rets:
            rets[Episode.ACTION_MASK] = f(rets[Episode.ACTION_MASK], np.vstack)

        return rets

    def seed(self, seed=None):
        self._env.seed(seed)

    def _build_from_base_dict(self, base_dict: Dict[AgentID, Any]):
        """Map the entry of a dict to a list with defined mapping function."""
        ans = {aid: [] for aid in self.possible_agents}
        for base_aid, data in base_dict.items():
            if base_aid == "__all__":  # (ziyu): This is for done['__all__']
                continue
            ans[self._ps_mapping_func(base_aid)].append(data)

        return ans

    def time_step(self, action_dict):
        base_action = self._extract_to_base(action_dict)
        rets = self._env.step(base_action)

        def f(x_dict, reduce_func):
            x_dict = self._build_from_base_dict(x_dict)
            return {k: reduce_func(v) for k, v in x_dict.items()}

        (obs, reward, done, info) = (
            f(rets[Episode.NEXT_OBS], np.vstack),
            f(rets[Episode.REWARD], np.vstack),
            f(rets[Episode.DONE], np.vstack),
            f(rets[Episode.INFO], lambda x: x[0]),
        )
        done["__all__"] = rets[Episode.DONE]["__all__"]

        for aid, action in action_dict.items():
            info[aid]["num_pass"] = np.logical_and(action <= 11, action >= 9).sum()
            info[aid]["num_shot"] = (action == 12).sum()

        res = {
            Episode.NEXT_OBS: obs,
            Episode.NEXT_STATE: self._build_state_from_obs(obs),
            Episode.REWARD: reward,
            Episode.DONE: done,
            Episode.INFO: info,
        }

        if Episode.ACTION_MASK in rets:
            res[Episode.ACTION_MASK] = f(rets[Episode.ACTION_MASK], np.vstack)

        return

    def env_done_check(self, agent_dones: Dict[AgentID, np.ndarray]) -> bool:
        return agent_dones["__all__"]

    def _extract_to_base(self, from_dict):
        to_dict = {}
        for aid, bucket in self._ps_buckets.items():
            for i, base_aid in enumerate(bucket):
                to_dict[base_aid] = from_dict[aid][i]
        return to_dict

    def close(self):
        self._env.close()

    def record_episode_info_step(self, rets):
        super().record_episode_info_step(rets)
        reward = rets[Episode.REWARD]
        info = rets[Episode.INFO]

        for aid in reward:
            self.custom_metrics[aid]["total_reward"] += reward[aid].sum()
            self.custom_metrics[aid]["win"] += info[aid].get("win", 0.0)
            self.custom_metrics[aid]["score"] += info[aid].get("score", 0.0)
            self.custom_metrics[aid]["goal_diff"] += info[aid].get("goal_diff", 0.0)
            self.custom_metrics[aid]["num_pass"] += info[aid]["num_pass"]
            self.custom_metrics[aid]["num_shot"] += info[aid]["num_shot"]
