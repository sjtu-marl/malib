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

from typing import Callable, Dict, Any, List, Tuple
from argparse import Namespace
from collections import defaultdict

import pytest
import ray

from malib.agent.agent_interface import AgentInterface
from malib.agent.manager import TrainingManager
from malib.backend.parameter_server import ParameterServer

# from malib.rollout.envs.dummy_env import env_desc_gen
from malib.runner import start_servers
from malib.rollout.envs.gym import env_desc_gen as gym_env_desc_gen
from malib.rollout.envs.open_spiel import env_desc_gen as open_spiel_env_desc_gen
from malib.rollout.envs.vector_env import VectorEnv
from malib.rollout.inference.utils import process_policy_outputs
from malib.rollout.rolloutworker import parse_rollout_info
from malib.utils.episode import Episode, NewEpisodeDict
from malib.utils.typing import AgentID, PolicyID
from malib.agent.indepdent_agent import IndependentAgent
from malib.common.strategy_spec import StrategySpec
from malib.scenarios.marl_scenario import MARLScenario
from malib.rollout.inference.ray.server import RayInferenceWorkerSet
from malib.rollout.inference.ray.client import env_runner, RayInferenceClient
from malib.utils.typing import BehaviorMode


def dqn():
    from malib.rl.dqn import DQNPolicy, DQNTrainer, DEFAULT_CONFIG

    algorithms = {
        "default": (
            DQNPolicy,
            DQNTrainer,
            # model configuration, None for default
            {
                "net_type": "general_net",
                "config": {"hidden_sizes": [64, 64]},
            },
            {},
        )
    }
    trainer_config = DEFAULT_CONFIG["training_config"].copy()
    return [algorithms, trainer_config]


def build_marl_scenario(
    algorithms: Dict[str, Dict],
    env_description: Dict[str, Any],
    learner_cls,
    trainer_config: Dict[str, Any],
    agent_mapping_func: Callable,
    runtime_logdir: str,
) -> MARLScenario:
    training_config = {
        "type": learner_cls,
        "trainer_config": trainer_config,
        "custom_config": {},
    }
    rollout_config = {
        "fragment_length": 200,  # every thread
        "max_step": 20,
        "num_eval_episodes": 10,
        "num_threads": 2,
        "num_env_per_thread": 2,
        "num_eval_threads": 1,
        "use_subproc_env": False,
        "batch_mode": "time_step",
        "postprocessor_types": ["defaults"],
        # every # rollout epoch run evaluation.
        "eval_interval": 1,
        "inference_server": "ray",  # three kinds of inference server: `local`, `pipe` and `ray`
    }
    scenario = MARLScenario(
        name="test_ray_inference",
        log_dir=runtime_logdir,
        algorithms=algorithms,
        env_description=env_description,
        training_config=training_config,
        rollout_config=rollout_config,
        agent_mapping_func=agent_mapping_func,
        stopping_conditions={
            "training": {"max_iteration": int(1e10)},
            "rollout": {"max_iteration": 1000, "minimum_reward_improvement": 1.0},
        },
    )
    return scenario


def push_policy_to_parameter_server(
    scenario: MARLScenario, parameter_server: ParameterServer
) -> Dict[AgentID, StrategySpec]:
    """Generate a dict of strategy spec, generate policies and push them to the remote parameter server.

    Args:
        scenario (MARLScenario): Scenario instance.
        agents (List[AgentID]): A list of enviornment agents.
        parameter_server (ParameterServer): Remote parameter server.

    Returns:
        Dict[AgentID, StrategySpec]: A dict of strategy specs.
    """

    res = dict()
    for agent in scenario.env_desc["possible_agents"]:
        sid = scenario.agent_mapping_func(agent)
        if sid in res:
            continue
        spec_pid = f"policy-0"
        strategy_spec = StrategySpec(
            identifier=sid,
            policy_ids=[spec_pid],
            meta_data={
                "policy_cls": scenario.algorithms["default"][0],
                "experiment_tag": "test_ray_inference",
                "kwargs": {
                    "observation_space": scenario.env_desc["observation_spaces"][agent],
                    "action_space": scenario.env_desc["action_spaces"][agent],
                    "model_config": scenario.algorithms["default"][2],
                    "custom_config": scenario.algorithms["default"][3],
                    "kwargs": {},
                },
            },
        )
        policy = strategy_spec.gen_policy()
        ray.get(parameter_server.create_table.remote(strategy_spec))
        ray.get(
            parameter_server.set_weights.remote(
                spec_id=strategy_spec.id,
                spec_policy_id=spec_pid,
                state_dict=policy.state_dict(),
            )
        )
        res[sid] = strategy_spec
    return res


def generate_cs(
    scenario: MARLScenario, dataset_server, parameter_server
) -> Tuple[RayInferenceClient, Dict[str, RayInferenceWorkerSet]]:
    env_desc = scenario.env_desc
    observation_spaces = env_desc["observation_spaces"]
    action_spaces = env_desc["action_spaces"]
    servers = dict.fromkeys(env_desc["possible_agents"], None)
    agent_group = defaultdict(list)
    for agent in env_desc["possible_agents"]:
        sid = scenario.agent_mapping_func(agent)
        agent_group[sid].append(agent)

    client = RayInferenceClient(
        env_desc=scenario.env_desc,
        dataset_server=dataset_server,
        max_env_num=scenario.rollout_config["num_env_per_thread"],
        use_subproc_env=scenario.rollout_config["use_subproc_env"],
        batch_mode=scenario.rollout_config["batch_mode"],
        postprocessor_types=scenario.rollout_config["postprocessor_types"],
        training_agent_mapping=scenario.agent_mapping_func,
    )

    for sid, agents in agent_group.items():
        servers[sid] = RayInferenceWorkerSet(
            agent_id=sid,
            observation_space=observation_spaces[agent],
            action_space=action_spaces[agent],
            parameter_server=parameter_server,
            governed_agents=agents.copy(),
        )

    return client, servers


from malib.rollout.inference.ray.client import process_env_rets


def rollout_func(
    episode_dict: NewEpisodeDict,
    client: RayInferenceClient,
    servers: Dict[str, RayInferenceWorkerSet],
    rollout_config,
    server_runtime_config,
    evaluate,
):
    env_rets = client.env.reset(
        fragment_length=rollout_config["fragment_length"],
        max_step=rollout_config["max_step"],
    )
    processed_env_ret, dataframes = process_env_rets(
        env_rets,
        preprocessor=server_runtime_config["preprocessor"],
        preset_meta_data={"evaluate": evaluate},
    )
    if episode_dict is not None:
        episode_dict.record(processed_env_ret, agent_first=False)

    cnt = 0
    while not client.env.is_terminated():
        grouped_dataframes = defaultdict(list)
        for agent, dataframe in dataframes.items():
            runtime_id = client.training_agent_mapping(agent)
            grouped_dataframes[runtime_id].append(dataframe)

        policy_outputs = {
            rid: server.compute_action(
                grouped_dataframes[rid], runtime_config=server_runtime_config
            )
            for rid, server in servers.items()
        }

        env_actions, processed_policy_outputs = process_policy_outputs(
            policy_outputs, client.env
        )

        assert len(env_actions) > 0, "inference server may be stucked."

        if episode_dict is not None:
            episode_dict.record(processed_policy_outputs, agent_first=True)

        env_rets = client.env.step(env_actions)
        if len(env_rets) < 1:
            dataframes = {}
            continue

        processed_env_ret, dataframes = process_env_rets(
            env_rets,
            preprocessor=server_runtime_config["preprocessor"],
            preset_meta_data={"evaluate": evaluate},
        )

        if episode_dict is not None:
            episode_dict.record(processed_env_ret, agent_first=False)

        cnt += 1


def data_servers():
    if not ray.is_initialized():
        ray.init()

    parameter_server, offline_dataset_server = start_servers()
    return parameter_server, offline_dataset_server


@pytest.mark.parametrize(
    "env_desc",
    [
        gym_env_desc_gen(env_id="CartPole-v1"),
        # open_spiel_env_desc_gen(env_id="kuhn_poker"),
        # mdp_env_desc_gen(env_id="two_round_dmdp"),
    ],
)
@pytest.mark.parametrize("learner_cls", [IndependentAgent])
@pytest.mark.parametrize("algorithms,trainer_config", [dqn()])
def test_inference_mechanism(env_desc, learner_cls, algorithms, trainer_config):
    parameter_server, dataset_server = data_servers()
    scenario: MARLScenario = build_marl_scenario(
        algorithms,
        env_desc,
        learner_cls,
        trainer_config,
        agent_mapping_func=lambda agent: agent,
        runtime_logdir="./logs",
    )
    client, servers = generate_cs(scenario, dataset_server, parameter_server)
    training_manager = TrainingManager(
        experiment_tag=scenario.name,
        stopping_conditions=scenario.stopping_conditions,
        algorithms=scenario.algorithms,
        env_desc=scenario.env_desc,
        agent_mapping_func=scenario.agent_mapping_func,
        training_config=scenario.training_config,
        log_dir=scenario.log_dir,
        remote_mode=True,
        resource_config=scenario.resource_config["training"],
        verbose=True,
    )
    data_entrypoints = {k: k for k in training_manager.agent_groups.keys()}

    # add policies and start training
    strategy_specs = training_manager.add_policies(n=scenario.num_policy_each_interface)
    strategy_specs = strategy_specs
    data_entrypoints = data_entrypoints

    rollout_config = scenario.rollout_config.copy()
    rollout_config["flag"] = "rollout"

    server_runtime_config = {
        "strategy_specs": strategy_specs,
        "behavior_mode": BehaviorMode.EXPLOITATION,
        "preprocessor": client.preprocessor,
    }

    dwriter_info_dict = dict.fromkeys(data_entrypoints.keys(), None)

    for rid, identifier in data_entrypoints.items():
        queue_id, queue = ray.get(
            dataset_server.start_producer_pipe.remote(name=identifier)
        )
        dwriter_info_dict[rid] = (queue_id, queue)

    eval_results, performance_results = env_runner(
        client,
        servers,
        rollout_config,
        server_runtime_config,
        dwriter_info_dict,
    )
    eval_results = parse_rollout_info([{"evaluation": eval_results}])
    print(eval_results["evaluation"])
    print(performance_results)

    for rid, identifier in data_entrypoints.items():
        ray.get(dataset_server.end_producer_pipe.remote(identifier))

    ray.kill(parameter_server)
    ray.kill(dataset_server)
    ray.shutdown()


# def test_inference_pipeline(self):
#     """This function tests the inference pipeline without using default env runner"""

#     training_manager.run(data_entrypoints)

#     rollout_config = scenario.rollout_config.copy()
#     rollout_config["flag"] = "rollout"
#     server_runtime_config = {
#         "strategy_specs": strategy_specs,
#         "behavior_mode": BehaviorMode.EXPLOITATION,
#         "preprocessor": client.preprocessor,
#     }

#     dwriter_info_dict = dict.fromkeys(data_entrypoints.keys(), None)

#     for rid, identifier in data_entrypoints.items():
#         queue_id, queue = ray.get(
#             dataset_server.start_producer_pipe.remote(name=identifier)
#         )
#         dwriter_info_dict[rid] = (queue_id, queue)

#     # collect episodes and run training
#     rollout_env = client.env
#     for n_epoch in range(2):
#         episode_dict = NewEpisodeDict(
#             lambda: Episode(agents=scenario.env_desc["possible_agents"])
#         )
#         rollout_func(
#             episode_dict,
#             client,
#             servers,
#             rollout_config,
#             server_runtime_config,
#             False,
#         )

#         episodes = episode_dict.to_numpy()
#         for rid, writer_info in dwriter_info_dict.items():
#             agents = client.agent_group[rid]
#             batches = []
#             for episode in episodes.values():
#                 agent_buffer = [episode[aid] for aid in agents]
#                 batches.append(agent_buffer)
#             writer_info[-1].put_nowait_batch(batches)
#         rollout_info = client.env.collect_info()
#         eval_results = list(rollout_info.values())
#         rollout_res = parse_rollout_info([{"evaluation": eval_results}])

#         print("epoch: {}\nrollout_res: {}\n".format(n_epoch, rollout_res))

#         client.env = rollout_env

#     training_manager.cancel_pending_tasks()
#     # training_manager.terminate()
