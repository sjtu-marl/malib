# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

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

from typing import List, Dict, Any, Union

import pytest
import numpy as np
import gym

from gym import spaces

from malib.utils.typing import AgentID
from malib.utils.episode import Episode
from malib.rollout.envs.env import Environment


class DummyEnv(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        self.num_agents = configs.get("num_agents", 2)
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3, 2))
        self.step_time_consumption = configs.get("step_time_consumption", 0.1)
        self.enable_env_state = configs.get("enable_env_state", False)

    @property
    def possible_agents(self) -> List[AgentID]:
        return [f"dummy_{i}" for i in range(self.num_agents)]

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return {agent: self.action_space for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return {agent: self.observation_space for agent in self.possible_agents}

    def time_step(self, actions: Dict[AgentID, Any]):
        observations = {
            agent: obs_space.sample()
            for agent, obs_space in self.observation_spaces.items()
        }
        states = (
            {
                agent: obs_space.sample()
                for agent, obs_space in self.observation_spaces.items()
            }
            if self.enable_env_state
            else None
        )
        rewards = {agent: 0.0 for agent in self.possible_agents}
        done = self.cnt >= self.max_step
        dones = dict.fromkeys(self.possible_agents, done)
        infos = dict.fromkeys(self.possible_agents, {})
        # fake time consumption
        # time.sleep(self.step_time_consumption)
        return states, observations, rewards, dones, infos

    def close(self):
        pass

    def reset(self, max_step: int = None) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super().reset(max_step)
        observations = {
            agent: obs_space.sample()
            for agent, obs_space in self.observation_spaces.items()
        }
        state = (
            {
                agent: obs_space.sample()
                for agent, obs_space in self.observation_spaces.items()
            }
            if self.enable_env_state
            else None
        )
        return state, observations


def generate_a_transition(batch: int):
    return {
        Episode.CUR_OBS: np.random.random((batch, 2)),
        Episode.ACTION: np.random.random((batch,)),
        Episode.REWARD: np.random.random((batch,)),
    }


class TestEpisode:
    """
    Episode is responsible for the trajectory collection of a group of agents. We need to ensure that it supports two kinds of gait situations.

    1. All agents have the same episode length: this is an usual case;
    2. There is a difference between agents in the length of timestep.
    """

    @pytest.fixture(autouse=True)
    def init(self):
        self.env = DummyEnv(enable_env_state=True)
        self.episode = Episode(agents=self.env.possible_agents)

    def test_assignment_and_retrive(self):
        """This function tests the `getter` and `setter`"""

        # supose we write a transition to each agent, then we should retrive the same one for each agent
        x = {k: generate_a_transition(2) for k in self.env.possible_agents}
        for agent in self.env.possible_agents:
            self.episode[agent] = x[agent]

        for k, v in self.episode.agent_entry.items():
            assert v == x[k]

    def test_record(self):
        state, obs = self.env.reset()

        done = False
        while not done:
            actions = {
                k: action_space.sample()
                for k, action_space in self.env.action_spaces.items()
            }
            self.episode.record(
                {Episode.CUR_OBS: obs, Episode.CUR_STATE: state}, agent_first=False
            )
            state, obs, rew, dones, info = self.env.step(actions)
            done = dones.pop("__all__")
            self.episode.record(
                {Episode.REWARD: rew, Episode.DONE: dones, Episode.ACTION: actions},
                agent_first=False,
            )
        self.episode.record(
            {Episode.CUR_OBS: obs, Episode.CUR_STATE: state}, agent_first=False
        )

        for agent, agent_trans in self.episode.agent_entry.items():
            for k, v in agent_trans.items():
                if k in [Episode.CUR_OBS, Episode.CUR_STATE, Episode.ACTION_MASK]:
                    assert len(v) == self.env.max_step + 1, (len(v), self.env.max_step)
                else:
                    assert len(v) == self.env.max_step, (len(v), self.env.max_step)

        for agent, agent_trans in self.episode.to_numpy().items():
            for k, v in agent_trans.items():
                print(k, v.shape)
                assert v.shape[0] == self.env.max_step, (v.shape, self.env.max_step)

    def test_ordering(self):
        episode = Episode(["agent"])
        actions = np.arange(100)
        observations = np.arange(101)
        rewards = np.arange(100)
        dones = np.zeros(100).astype(np.bool)
        terminal_points = np.random.choice(100, 3, replace=False)
        dones[terminal_points] = True

        labels = {
            Episode.CUR_OBS: observations[:100],
            Episode.NEXT_OBS: observations[1:],
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            Episode.ACTION: actions,
        }

        for cnt in range(100):
            obs = observations[cnt]
            episode.record({Episode.CUR_OBS: {"agent": obs}}, agent_first=False)
            rew, done = rewards[cnt], dones[cnt]
            action = actions[cnt]
            episode.record(
                {
                    Episode.REWARD: {"agent": rew},
                    Episode.DONE: {"agent": done},
                    Episode.ACTION: {"agent": action},
                },
                agent_first=False,
            )

        episode.record(
            {Episode.CUR_OBS: {"agent": observations[cnt + 1]}}, agent_first=False
        )

        for k, v in episode.to_numpy()["agent"].items():
            assert np.all(np.equal(labels[k], v)), (k, v.shape, labels[k].shape)

    # def test_data_ordering(self):
    #     import ray
    #     import torch
    #     import gym

    #     from malib.backend.offline_dataset_server import OfflineDataset
    #     from malib.rl.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG
    #     from malib.rollout.envs.gym import GymEnv

    #     if not ray.is_initialized():
    #         ray.init(num_cpus=2)

    #     dataset_server: OfflineDataset = OfflineDataset.as_remote().remote(
    #         table_capacity=10000
    #     )
    #     _, writer = ray.get(dataset_server.start_producer_pipe.remote("test"))
    #     _, reader = ray.get(
    #         dataset_server.start_consumer_pipe.remote(
    #             "test", DEFAULT_CONFIG["training_config"]["batch_size"]
    #         )
    #     )

    #     env = GymEnv(env_id="CartPole-v1")
    #     policy = DQNPolicy(
    #         observation_space=env.observation_spaces["agent"],
    #         action_space=env.action_spaces["agent"],
    #         model_config=DEFAULT_CONFIG["model_config"],
    #         custom_config=DEFAULT_CONFIG["custom_config"],
    #     )
    #     trainer = DQNTrainer(
    #         training_config=DEFAULT_CONFIG["training_config"], policy_instance=policy
    #     )

    #     num_episode = 10000
    #     cnt = 0

    #     for _ in range(num_episode):
    #         done = False

    #         episode = Episode(agents=["agent"])
    #         state, obs = env.reset(max_step=200)

    #         episode.record({Episode.CUR_OBS: obs}, agent_first=False)

    #         while not done:
    #             cnt += 1

    #             obs = {
    #                 k: torch.from_numpy(v).float().reshape(1, -1)
    #                 for k, v in obs.items()
    #             }
    #             actions = {
    #                 k: policy.compute_action(v, None, evaluate=False)[0][0]
    #                 for k, v in obs.items()
    #             }
    #             state, obs, rew, dones, info = env.step(actions)

    #             done = dones.pop("__all__")
    #             episode.record(
    #                 {
    #                     Episode.ACTION: actions,
    #                     Episode.CUR_OBS: obs,
    #                     Episode.REWARD: rew,
    #                     Episode.DONE: dones,
    #                 },
    #                 agent_first=False,
    #             )

    #             if cnt % 20 == 0:
    #                 batch_info = reader.get()
    #                 if len(batch_info[-1]) == 0:
    #                     continue
    #                 loss_info = trainer(batch_info[0])

    #         agent_numpy = episode.to_numpy()
    #         writer.put_nowait_batch([agent_numpy["agent"]])

    #         # run eval
    #         mean_episode_rew = 0.0
    #         mean_step_length = 0.0

    #         for _ in range(5):
    #             done = False
    #             episode_rew = 0.0
    #             eval_length = 0
    #             state, obs = env.reset(max_step=200)
    #             while not done:
    #                 eval_length += 1
    #                 obs = {
    #                     k: torch.from_numpy(v).float().reshape(1, -1)
    #                     for k, v in obs.items()
    #                 }
    #                 actions = {
    #                     k: policy.compute_action(v, None, evaluate=True)[0][0]
    #                     for k, v in obs.items()
    #                 }
    #                 state, obs, rew, dones, info = env.step(actions)
    #                 done = dones.pop("__all__")
    #                 episode_rew += rew["agent"]
    #             mean_episode_rew += episode_rew / 5
    #             mean_step_length += eval_length / 5

    #         print("mean_episode_rew:", mean_episode_rew, mean_step_length, policy.eps)
