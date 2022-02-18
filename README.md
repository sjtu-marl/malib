
<div align=center><img src="docs/imgs/logo.svg" width="35%"></div>


# MALib: A parallel framework for population-based multi-agent reinforcement learning

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sjtu-marl/malib/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest)

MALib is a parallel framework of population-based learning nested with (multi-agent) reinforcement learning (RL) methods, such as Policy Space Response Oracle, Self-Play and Neural Fictitious Self-Play. MALib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms. The design of MALib also strives to promote the research of other multi-agent learning, including multi-agent imitation learning and model-based MARL.

![architecture](docs/imgs/Architecture.svg)

## Installation

The installation of MALib is very easy. We've tested MALib on Python 3.6 and 3.7. This guide is based on ubuntu 18.04 and above. We strongly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.


```bash
conda create -n malib python==3.7 -y
conda activate malib

# install dependencies
./install_deps.sh

# install malib
pip install -e .
```

External environments are integrated in MALib, such as StarCraftII and vizdoom, you can install them via `pip install -e .[envs]`. For users who wanna contribute to our repository, run `pip install -e .[dev]` to complete the development dependencies.

**NOTE**: if you wanna use alpha-rank (default solver for meta game) to solve meta-game, install open-spiel with its [installation guides](https://github.com/deepmind/open_spiel)

## Quick Start

```python
"""PSRO with PPO for Leduc Holdem"""

from malib.envs.poker import poker_aec_env as leduc_holdem
from malib.runner import run
from malib.rollout import rollout_func


env_description = {
    "creator": PokerParallelEnv,
    "config": {
        "scenario_configs": {"fixed_player": True},
        "env_id": "leduc_poker",
    },
}
env = PokerParallelEnv(**env_description["config"])
possible_agents = env.possible_agents
observation_spaces = env.observation_spaces
action_spaces = env.action_spaces

env_description["possible_agents"] = possible_agents
env_description.update(
    {
        "action_spaces": env.action_spaces,
        "observation_spaces": env.observation_spaces,
    }
)

run(
    group="psro",
    name="leduc_poker",
    env_description=env_description,
    training={
        "interface": {
            "type": "independent",
            "observation_spaces": observation_spaces,
            "action_spaces": action_spaces,
            "use_init_policy_pool": True,
        },
        "config": {
            "batch_size": args.batch_size,
            "update_interval": args.num_epoch,
        },
    },
    algorithms={
        args.algorithm: {
            "name": args.algorithm,
            "custom_config": {
                "gamma": 1.0,
                "eps_min": 0,
                "eps_max": 1.0,
                "eps_anneal_time": 100,
                "lr": 1e-2,
            },
        }
    },
    rollout={
        "type": "async",
        "stopper": "simple_rollout",
        "stopper_config": {"max_step": 1000},
        "metric_type": "simple",
        "fragment_length": 100,
        "num_episodes": 1,
        "num_env_per_worker": 1,
        "max_step": 10,
        "postprocessor_types": ["copy_next_frame"],
    },
    evaluation={
        "max_episode_length": 100,
        "num_episode": 100,
    },
    global_evaluator={
        "name": "psro",
        "config": {
            "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
        },
    },
    dataset_config={"episode_capacity": 200000},
    task_mode="gt",
)
```

## Documentation

See [MALib Docs](https://malib.readthedocs.io/)

## Citing MALib


If you use MALib in your work, please cite the accompanying [paper](https://arxiv.org/abs/2106.07551).

```bibtex
@misc{zhou2021malib,
      title={MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning}, 
      author={Ming Zhou and Ziyu Wan and Hanjing Wang and Muning Wen and Runzhe Wu and Ying Wen and Yaodong Yang and Weinan Zhang and Jun Wang},
      year={2021},
      eprint={2106.07551},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
