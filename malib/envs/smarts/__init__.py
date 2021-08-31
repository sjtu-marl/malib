# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from pathlib import Path

import gym
import re
import yaml

from pathlib import Path
from malib.envs.smarts.agents import load_config
from malib.envs.smarts._env.smarts.core.agent import AgentSpec
from malib.envs.smarts._env.smarts.core.scenario import Scenario
from malib.envs.smarts._env.smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    NeighborhoodVehicles,
    Waypoints,
)
from malib.envs.smarts._env.smarts.core.controllers import ActionSpaceType
from . import common


def _make_config(config):
    """Generate agent configuration. `mode` can be `train` or 'evaluation', and the
    only difference on the generated configuration is the agent info adapter.
    """

    agent_config = config["agent"]
    interface_config = config["interface"]

    """ Parse the state configuration for agent """
    state_config = agent_config["state"]

    # initialize environment wrapper if the wrapper config is not None
    wrapper_config = state_config.get("wrapper", {"name": "Simple"})
    features_config = state_config["features"]
    # only for one frame, not really an observation
    frame_space = gym.spaces.Dict(common.subscribe_features(**features_config))
    action_type = ActionSpaceType(agent_config["action"]["type"])
    env_action_space = common.ActionSpace.from_type(action_type)

    """ Parse policy configuration """
    policy_obs_space = frame_space
    policy_action_space = env_action_space

    observation_adapter = common.ObservationAdapter(frame_space, features_config)
    # wrapper_cls.get_observation_adapter(
    #     policy_obs_space, feature_configs=features_config, wrapper_config=wrapper_config
    # )
    action_adapter = common.ActionAdapter.from_type(action_type)
    # policy observation space is related to the wrapper usage
    # policy_config = (
    #     None,
    #     policy_obs_space,
    #     policy_action_space,
    #     config["policy"].get(
    #         "config", {"custom_preprocessor": wrapper_cls.get_preprocessor()}
    #     ),
    # )

    """ Parse agent interface configuration """
    if interface_config.get("neighborhood_vehicles"):
        interface_config["neighborhood_vehicles"] = NeighborhoodVehicles(
            **interface_config["neighborhood_vehicles"]
        )

    if interface_config.get("waypoints"):
        interface_config["waypoints"] = Waypoints(**interface_config["waypoints"])

    if interface_config.get("rgb"):
        interface_config["rgb"] = RGB(**interface_config["rgb"])

    if interface_config.get("ogm"):
        interface_config["ogm"] = OGM(**interface_config["ogm"])

    interface_config["action"] = ActionSpaceType(action_type)

    """ Pack environment configuration """
    config["env_config"] = {
        "custom_config": {
            **wrapper_config,
            "reward_adapter": lambda x: x,
            "observation_adapter": observation_adapter,
            "action_adapter": action_adapter,
            "observation_space": policy_obs_space,
            "action_space": policy_action_space,
        },
    }
    config["agent"] = {"interface": AgentInterface(**interface_config)}
    # config["trainer"] = _get_trainer(**config["policy"]["trainer"])
    # config["policy"] = policy_config

    # print(format.pretty_dict(config))

    return config


def load_config(config_file):
    """Load algorithm configuration from yaml file.

    This function support algorithm implemented with RLlib.
    """
    yaml.SafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    base_dir = Path(__file__).absolute().parent.parent.parent
    with open(base_dir / config_file, "r") as f:
        raw_config = yaml.safe_load(f)
    return _make_config(raw_config)


def gen_config(**kwargs):
    scenario_path = Path(kwargs["scenario"]).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(kwargs["config_file"])
    agents = {agent_id: AgentSpec(**config["agent"]) for agent_id in agent_ids}

    config["env_config"].update(
        {
            "seed": 42,
            "scenarios": [str(scenario_path)],
            "headless": kwargs["headless"],
            "agent_specs": agents,
        }
    )

    # if kwargs["paradigm"] == "centralized":
    #     config["env_config"].update(
    #         {
    #             "obs_space": gym.spaces.Tuple([obs_space] * agent_missions_count),
    #             "act_space": gym.spaces.Tuple([act_space] * agent_missions_count),
    #             "groups": {"group": agent_ids},
    #         }
    #     )
    #     tune_config.update(config["policy"][-1])
    # else:
    #     policies = {}
    #     for k in agents:
    #         policies[k] = config["policy"][:-1] + (
    #             {**config["policy"][-1], "agent_id": k},
    #         )
    #     tune_config.update(
    #         {
    #             "multiagent": {
    #                 "policies": policies,
    #                 "policy_mapping_fn": lambda agent_id: agent_id,
    #             }
    #         }
    #     )

    return config
