"""
Wrapper of VizDoom

State:
    - number
    - game_variables
    - screen_buffer
    - depth_buffer
    - labels_buffer
    - automap_buffer
    - labels
    - objects
    - sectors
"""
import os.path
import time

import gym
import random

import numpy as np
from pettingzoo import ParallelEnv
from pprint import pprint

# import skimage.transform
import vizdoom as vzd

from malib.utils.typing import AgentID, Dict, Any, Tuple


def make_env(
    doom_scenario_path: str,
    doom_map: str,
    mode=vzd.Mode.ASYNC_PLAYER,
    screen_resolution=vzd.ScreenResolution.RES_160X120,
    screen_format=vzd.ScreenFormat.GRAY8,  # vzd.ScreenFormat.RGB24,
    depth_buffer_enabled=False,
    labels_buffer_enabled=False,
    automap_buffer_enabled=False,
    objects_info_enabled=False,
    sectors_info_enabled=False,
    window_visible=False,  # Makes the window appear (turned on by default)
    render_hud=False,
    render_minimal_hud=False,
    render_crosshair=False,
    render_weapon=True,
    render_decals=False,
    render_particles=False,
    render_effects_sprites=False,
    render_messages=False,
    render_screen_flashes=False,
    living_reward=-1,
    episode_timeout=200,  # Makes episodes start after 10 tics (~after raising the weapon)
    episode_start_time=10,  # Causes episodes to finish after 200 tics (actions)
    sound_enabled=False,  # Turns off the sound. (turned off by default)
):
    params = locals()
    env = raw_env(**params)
    # env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """To support the AEC API, the raw_env() function just uses the from_parallel function to convert from
    a ParallelEnv to an AEC env.
    """
    env = parallel_env(**kwargs)
    # env = from_parallel(env)
    return env


def state_transform(state: vzd.GameState, resolution=(640, 480)):
    """Return only image (gray 8)"""

    # return state
    if state is None:
        return None
    else:
        return np.expand_dims(state.screen_buffer, axis=0) / 255
    # if state is None:
    #     return None
    # else:
    #     img = skimage.transform.resize(state.screen_buffer, resolution)
    #     img = img.astype(np.float32)
    #     img = np.expand_dims(img, axis=0)
    #     return img


def action_transform(actions: Dict[AgentID, int], action_dim: int):
    res = {agent: [False] * action_dim for agent in actions}
    for agent, a in actions.items():
        res[agent][a] = True
    return res


# maximum of episode length
NUM_ITERS = int(1e10)
SKIP_RATE = 4
FRAME_REPEAT = SKIP_RATE


class parallel_env(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "vizdoom_v1"}

    def __init__(self, **kwargs):
        self.possible_agents = [f"doom_{i}" for i in range(kwargs.get("num_agents", 1))]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # 3 actions mapping to: move_left, move_right, attack
        self.action_spaces: Dict[AgentID, gym.spaces.Space] = {
            agent: gym.spaces.Discrete(3) for agent in self.possible_agents
        }
        self.observation_spaces: Dict[AgentID, gym.spaces.Space] = {
            agent: gym.spaces.Box(low=0.0, high=1.0, shape=(1, 160, 120))
            for agent in self.possible_agents
        }  # {agent: None for agent in self.possible_agents}
        self._sleep_time = 0.0  # 1.0 / vzd.DEFAULT_TICRATE  # = 0.028 for render
        self.game = vzd.DoomGame()
        self.setup(**kwargs)

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            # for set
            if v is None:
                continue
            getattr(self.game, f"set_{k}")(v)

        game = self.game

        game.add_available_button(vzd.Button.MOVE_LEFT)
        game.add_available_button(vzd.Button.MOVE_RIGHT)
        game.add_available_button(vzd.Button.ATTACK)
        # game.add_available_game_variable(vzd.GameVariable.AMMO2)
        game.init()

    def render(self, mode: str = "human") -> None:
        """Renders the environment. In human mode, it can print to terminal, open up a graphical window,
        or open up some other display that a human can see and understand.

        :param str mode: Visualization mode. Default to human.
        :return: None
        """
        raise NotImplementedError

    def reset(self):
        """Reset needs to initialize the `agents` attributes and must set up the environment so that render(), and step()
        can be called without issues.

        :return: A dictionary of agent observations
        """

        self.agents = self.possible_agents[:]
        self.num_moves = 0

        # Starts a new episode. It is not needed right after init() but it doesn't cost much.
        # At least the loop is nicer.
        self.game.new_episode()
        # FIXME(ming): reset game core
        self.game.get_state()

        # if self._sleep_time > 0:
        #     time.sleep(self._sleep_time)

        observations = {
            agent: state_transform(self.game.get_state()) for agent in self.agents
        }
        return observations

    def close(self):
        self.game.close()

    def get_total_reward(self):
        return self.game.get_total_reward()

    def step(self, actions: Dict[AgentID, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Environment stepping by taking agent actions and return: `observations`, `rewards`, `dones` and `infos`. Dicts
        where each dict looks lke {agent_1: item_1, agent_2: item_2}.

        :param Dict[AgentID,Any] actions: A dict of agent actions.
        :return: A tuple of environment returns.
        """

        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        actions = action_transform(actions, 3)
        rewards = {
            agent: self.game.make_action(actions[agent], FRAME_REPEAT)
            for agent in self.agents
        }
        self.num_moves += 1

        env_done = self.num_moves >= NUM_ITERS or self.game.is_episode_finished()
        dones = {agent: env_done for agent in self.agents}
        dones["__all__"] = any(dones.values())
        observations = {
            agent: state_transform(
                self.game.get_state(), resolution=self.observation_spaces[agent].shape
            )
            for agent in self.agents
        }
        infos = {
            agent: {
                "living_reward": self.game.get_living_reward(),
                "last_reward": self.game.get_last_reward(),
                "last_action": self.game.get_last_action(),
                "available_action": self.game.get_available_buttons(),
                "step": self.num_moves,
            }
            for agent in self.agents
        }

        return observations, rewards, dones, infos


def meta_info(data):
    return {
        "type": type(data),
        "shape": data.shape if hasattr(data, "shape") else "No shape",
        "agg_sum_value": np.sum(data) if isinstance(data, np.ndarray) else data,
        "agg_mean_value": np.mean(data) if isinstance(data, np.ndarray) else data,
        "agg_var_value": np.var(data) if isinstance(data, np.ndarray) else data,
    }


def parse_state(state: vzd.GameState):
    if state is None or not isinstance(state, vzd.GameState):
        return state
    else:
        return {
            "time": meta_info(state.number),
            "vars": meta_info(state.game_variables),
            "screen_buf": meta_info(state.screen_buffer),
            "depth_buf": meta_info(state.depth_buffer),
            "labels_buf": meta_info(state.labels_buffer),
            "automap_buf": meta_info(state.automap_buffer),
            "labels": meta_info(state.labels),
            "objects": meta_info(state.objects),
            "sectors": meta_info(state.sectors),
        }


if __name__ == "__main__":
    env = make_env(
        doom_scenario_path=os.path.join(vzd.scenarios_path, "basic.wad"),
        doom_map="map01",
    )

    agents = env.possible_agents
    obs = env.reset()
    done = False

    iter = 0
    while not done:
        actions = {agent: random.choice([0, 1, 2]) for agent in agents}
        observations, rewards, dones, infos = env.step(actions)
        print(f"=================\nstep on #{iter}:")
        parsed_state = {agent: parse_state(v) for agent, v in observations.items()}
        print("game state:")
        pprint(parsed_state)
        pprint(f"reward: {rewards}")
        done = dones["__all__"]
        print("==================")
        iter += 1

    print("Episode finished")
    print(f"Total reward: {env.get_total_reward()}")
    print("********************")
    env.close()
