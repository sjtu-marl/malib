# MALib
A general-purpose multi-agent training framework.

## Installation

**step1: build environment**

```shell
conda create -n malib python==3.7 -y
conda activate malib
pip install -e .

# for development
pip install -e .[dev]
```
**step2: install openspiel**

installation guides: [openspiel](https://github.com/deepmind/open_spiel)

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