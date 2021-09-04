import os.path as osp
import gym
import re
import yaml

from pathlib import Path

from malib.envs import Environment, env
from malib.backend.datapool.offline_dataset_server import Episode

from malib.utils.typing import Dict, Sequence, Any, AgentID
from smarts.core.agent import AgentSpec
from smarts.core.scenario import Scenario
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from . import common


BASE_DIR = osp.dirname(osp.abspath(__file__))


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
    info_adapter = common.InfoAdapter
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
    config["agent"] = {
        "interface": AgentInterface(**interface_config),
        "observation_adapter": observation_adapter,
        "action_adapter": action_adapter,
        "info_adapter": info_adapter,
    }
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


def gen_config(
    scenarios: Sequence,
    agent_config_file: str,
    headless: bool = True,
):
    assert len(scenarios) == 1, "Accept only one scenarios now!"
    scenario_path = Path(scenarios[0]).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(agent_config_file)
    agents = {agent_id: AgentSpec(**config["agent"]) for agent_id in agent_ids}

    config["env_config"].update(
        {
            "seed": 42,
            "scenarios": [str(scenario_path)],
            "headless": headless,
            "agent_specs": agents,
        }
    )

    return config


class SMARTS(Environment):
    def __init__(self, **configs):
        super(SMARTS, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs: Dict[str, Any] = self._configs["scenario_configs"]

        scenario_paths = scenario_configs["path"]
        agent_type = scenario_configs["agent_type"]

        # generate abs paths
        scenario_paths = list(map(lambda x: osp.join(BASE_DIR, x), scenario_paths))
        max_step = scenario_configs["max_step"]

        parsed_configs = gen_config(
            scenarios=scenario_paths,
            agent_config_file=osp.join(BASE_DIR, "agents", agent_type),
        )
        env_config = parsed_configs["env_config"]
        custom_config = env_config["custom_config"]

        # build agent specs with agent interfaces
        self.is_sequential = False
        self.scenarios = scenario_paths
        self._env = gym.make(
            "smarts.env:hiway-v0",
            agent_specs=env_config["agent_specs"],
            headless=True,
            scenarios=scenario_paths,
        )
        self._env.possible_agents = list(self._env.agent_specs.keys())
        self._env.observation_spaces = {
            agent: custom_config["observation_space"]
            for agent in self._env.possible_agents
        }
        self._env.action_spaces = {
            agent: custom_config["action_space"] for agent in self._env.possible_agents
        }
        self._trainable_agents = self._env.possible_agents
        self._max_step = max_step

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(actions)
        # remove dones all
        dones.pop("__all__")
        super(SMARTS, self).step(actions, rewards=rewards, dones=dones, infos=infos)

        return {
            Episode.CUR_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            Episode.INFO: infos,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self, *args, **kwargs):
        observations = super(SMARTS, self).reset(*args, **kwargs)
        self._max_step = self._max_step or kwargs.get("max_step", None)
        return {Episode.CUR_OBS: observations}
