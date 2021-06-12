.. _quick-start:

Quick Start
===========

If you have not installed MALib yet, please refer to :ref:`installation` before running. We give two cases of running `Policy Space Response Oracle (PSRO) <https://arxiv.org/pdf/1711.00832.pdf>`_ to solve `Leduc Holdem <https://en.wikipedia.org/wiki/Texas_hold_%27em>`_, and `MADDPG <https://arxiv.org/abs/1706.02275>`_ to solve a particle multi-agent cooperative task, `Simple Spread <https://www.pettingzoo.ml/mpe/simple_spread>`_.


PSRO Learning
-------------

**Policy Space Response Oracle (PSRO)** is a population-based MARL algorithm which cooperates game-theory and MARL algorithm to solve multi-agent tasks in the scope of meta-game. At each iteration, the algorithm will generate some policy combinations and executes independent learning for each agent. Such a nested learning process comprises rollout, training, evaluation in sequence, and works circularly until the algorithm finds the estimated Nash Equilibrium. 

.. note:: If you want to use alpha\-rank to estimate the equilibrium, you need to install open\-spiel before that. Follow the :ref:`installation` to get more details.

**Specify the environment**: The first thing to start your training task is to determine the environment for policy learning. Here, we use the Leduc Hodlem environment as an example. If you want to use custom environment, please refer to the :ref:`api-environment-custom` to get more details.

.. code-block:: python

    from malib.envs.poker import poker_aec_env as leduc_holdem

    env = leduc_holdem.env(fixed_player=True)
    env_description = {
        "creator": leduc_holdem.env,
        "config": {"fixed_player": True},
        "id": "leduc_holdem",
        "possible_agents": env.possible_agents
    }


**Specify the training interface**: ...

.. code-block:: python

    training={
        "interface": {
            "type": "independent",
            "observation_spaces": env.observation_spaces,
            "action_spaces": env.action_spaces
        },
    },

**Specify the rollout interface**: specify the rollout interface in MALib is very simple, we've implemented several rollout interfaces to meet different distributed computing requirements. In this case, we use ``AsyncRollout`` to support the high...

.. code-block:: python

    from malib.rollout import rollout_func

    rollout = {
        "type": "async",
        "stopper": "simple_rollout",
        "callback": rollout_func.sequential
    }


**Specify the underlying (MA)RL algorithm**: PSRO requires an underlying RL algorithm to find the best response at each learning iteration, you need to specify the algorithm you want to use in this learning. As a standard implementation, the underlying algorithm is PPO.

.. code-block:: python

    algorithms = {
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


The completed distributed execution example is presented below.

.. code-block:: python

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


As you can see, the enter function ``run`` 

Multi-agent Reinforcement Learning
----------------------------------