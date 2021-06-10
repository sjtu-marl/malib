
# MALib: A parallel framework for population-based multi-agent reinforcement learning

MALib is a parallel framework of population-based learning nested with (multi-agent) reinforcement learning (RL) methods, such as Policy Space Response Oracle, Self-Play and Neural Fictitous Self-Play. MALib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms. The design of MALib also strives to promto the research of other multi-agent learning research, including multi-agent imitation learning and model-based RL.

## Installation

We've tested MALib on Python 3.6 and 3.7

```bash
conda create -n malib python==3.7 -y
conda activate malib
pip install -e .

# for development
pip install -e .[dev]
```

**optional**: if you wanna use alpha-rank to solve meta-game, install open-spiel with its [installation guides](https://github.com/deepmind/open_spiel)

## Quick Start

```python
"""PSRO with PPO for Leduc Holdem"""

from malib.envs.poker import poker_aec_env as leduc_holdem
from malib.runner import run
from malib.rollout import rollout_func


env = leduc_holdem.env(fixed_player=True)

run(
    agent_mapping_func=lambda agent_id: agent_id,
    env_description={
        "creator": leduc_holdem.env,
        "config": {"fixed_player": True},
        "id": "leduc_holdem",
        "possible_agents": env.possible_agents,
    },
    training={
        "interface": {
            "type": "independent",
            "observation_spaces": env.observation_spaces,
            "action_spaces": env.action_spaces
        },
    },
    algorithms={
        "PSRO_PPO": {
            "name": "PPO",
            "custom_config": {
                "gamma": 1.0,
                "eps_min": 0,
                "eps_max": 1.0,
                "eps_decay": 100,
            },
        }
    },
    rollout={
        "type": "async",
        "stopper": "simple_rollout",
        "callback": rollout_func.sequential
    }
)
```