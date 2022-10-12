from typing import Callable, Dict, Any, List
from argparse import Namespace
from collections import defaultdict

import pytest
import ray
from malib.backend.parameter_server import ParameterServer

from malib.rollout.envs.dummy_env import env_desc_gen
from malib.rollout.rolloutworker import parse_rollout_info
from malib.utils.typing import AgentID, PolicyID
from malib.agent.indepdent_agent import IndependentAgent
from malib.common.strategy_spec import StrategySpec
from malib.runner import start_servers
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
            {},
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
    trainer_config["total_timesteps"] = 1000
    training_config = {
        "type": learner_cls,
        "trainer_config": trainer_config,
        "custom_config": {},
    }
    rollout_config = {
        "fragment_length": 2000,  # every thread
        "max_step": 200,
        "num_eval_episodes": 10,
        "num_threads": 2,
        "num_env_per_thread": 10,
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


@pytest.mark.parametrize("env_desc", [env_desc_gen()])
@pytest.mark.parametrize("learner_cls", [IndependentAgent])
@pytest.mark.parametrize("algorithms,trainer_config", [dqn()])
def test_env_runner(env_desc, algorithms, trainer_config, learner_cls):
    try:
        start_ray_info = ray.init(address="auto")
    except ConnectionError:
        start_ray_info = ray.init(num_cpus=3)

    parameter_server, dataset_server = start_servers()
    scenario: MARLScenario = build_marl_scenario(
        algorithms,
        env_desc,
        learner_cls,
        trainer_config,
        agent_mapping_func=lambda agent: agent,
        runtime_logdir="./logs",
    )

    agents = env_desc["possible_agents"]
    observation_spaces = env_desc["observation_spaces"]
    action_spaces = env_desc["action_spaces"]

    # create non- parameter sharing / grouping C-S instances
    servers = dict.fromkeys(agents, None)
    agent_group = defaultdict(list)
    for agent in agents:
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

    strategy_specs = push_policy_to_parameter_server(scenario, parameter_server)
    rollout_config = scenario.rollout_config.copy()
    rollout_config["flag"] = "rollout"
    server_runtime_config = {
        "strategy_specs": strategy_specs,
        "behavior_mode": BehaviorMode.EXPLOITATION,
        "preprocessor": client.preprocessor,
    }

    data_entrypoints = {k: k for k in agent_group.keys()}
    dwriter_info_dict = dict.fromkeys(data_entrypoints.keys(), None)

    for rid, identifier in data_entrypoints.items():
        queue_id, queue = ray.get(
            dataset_server.start_producer_pipe.remote(name=identifier)
        )
        dwriter_info_dict[rid] = (queue_id, queue)

    eval_results, performance = env_runner(
        client,
        servers,
        rollout_config=rollout_config,
        server_runtime_config=server_runtime_config,
        dwriter_info_dict=dwriter_info_dict,
    )

    x = parse_rollout_info([{"evaluation": eval_results}])
    import pdb

    pdb.set_trace()
