import yaml
import os
import pickle
import numpy as np

from malib import settings
from malib.envs.gr_football import env_desc_gen
from malib.algorithm.mappo import MAPPO
from malib.utils.episode import EpisodeKey


yaml_path = os.path.join(
    settings.BASE_DIR, "examples/mappo_gfootball/mappo_5_vs_5.yaml"
)


with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

env_desc = env_desc_gen(config["env_description"]["config"])
env = env_desc["creator"](**env_desc["config"])

obs_spaces = env_desc["observation_spaces"]
act_spaces = env_desc["action_spaces"]
possible_agents = env_desc["possible_agents"]

agents = {}
model_config = config["algorithms"]["MAPPO"]["model_config"]
custom_config = config["algorithms"]["MAPPO"]["custom_config"]
model_path = os.path.join(
    settings.BASE_DIR,
    "logs/experiment/case_1647485877.9991865/models/team_0/MAPPO_0.pkl",
)

for aid in possible_agents:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    agents[
        aid
    ] = model  # MAPPO("mappo", obs_spaces[aid], act_spaces[aid], model_config={}, custom_config={}, env_agent_id=aid)
    # model.load_state


# observation, action_mask, state
# eval for 10 episodes
for i in range(10):
    rets = env.reset()
    done = (False,)
    cnt = 0

    rnn_state = {}
    action_prob = {}
    action = {}
    for aid, model in agents.items():
        rnn_state[aid] = model.get_initial_state(4)
        action[aid], action_prob[aid], rnn_state[aid] = model.compute_action(
            observation=rets["observation"][aid],
            state=rets["state"][aid],
            rnn_state=rnn_state,
            done=done,
            action_mask=rets["action_mask"][aid],
        )
    env.step(action)

    while not done and cnt < 3001:
        env.step()
