.. _environments-doc:

Environments
============

MALib implements an unified environment interface to satisfy both turn-based and simultaneous-based environments

Current MALib works different environment types including MDP environment, openai-gym, OpenSpiel and any other user-defined environment under MALib's environment API.
We first introduce the available environments supported by MALib and then give an example of how to customize your own environments.

Available Environments
----------------------

* MDP environments for the OpenAI Gym
mdp_ is a simple and easy-to-specify environment for standard Markov Decision Process. *In MALib, this environment is used as a minimal testbed for verification of our algorithms' implementation.*

..  _mdp: https://github.com/BlackHC/mdp 

* OpenAI-Gym: single-agent environments
Gym_ is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Since its release, Gym's API has become the field standard for doing this.

.. _Gym: https://github.com/openai/gym

* DeepMind OpenSpiel: for empirical game theoretical research
OpenSpiel_ is a collection of environments and algorithms for research in general reinforcement learning and search/planning in games. OpenSpiel supports n-player (single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and sequential, strictly turn-taking and simultaneous-move, perfect and imperfect information games, as well as traditional multiagent environments such as (partially- and fully- observable) grid worlds and social dilemmas. OpenSpiel also includes tools to analyze learning dynamics and other common evaluation metrics. Games are represented as procedural extensive-form games, with some natural extensions. 

.. _OpenSpiel: https://github.com/deepmind/open_spiel

Environment Customiztion
------------------------
MALib defines a specific class of `Environment` which is similar to `gym.Env` with some modifications to support multi-agent scenarios.

Environment Configuration
>>>>>>>>>>>>>>>>>>>>>>>>>

`Environment`'s interaction interface e.g. `step` and `reset` take a dict as input/output type in the form of `<AgentID, content>` pairs to inform MALib of different agents' state and action and rewards and etc..


.. code-block:: python

    class Environment:
        def __init__(self, **configs):
            """ build your environment with configs """
            pass

        def reset(self, max_step: int = None) -> Union[None, Sequence[Dict[AgentID, Any]]]:
            """Reset environment and the episode info handler here."""
            pass 

        def step(
            self, actions: Dict[AgentID, Any]
        ) -> Tuple[
            Dict[AgentID, Any],
            Dict[AgentID, Any],
            Dict[AgentID, float],
            Dict[AgentID, bool],
            Any,
        ]:
            """Stepping function"""
            pass

        @property
        def possible_agents(self) -> List[AgentID]:
            """Return a list of environment agent ids"""
            raise NotImplementedError

        @property
        def observation_spaces(self) -> Dict[AgentID, gym.Space]:
            """A dict of agent observation spaces"""
            raise NotImplementedError

        @property
        def action_spaces(self) -> Dict[AgentID, gym.Space]:
            """A dict of agent action spaces"""
            raise NotImplementedError

MALib also supports `Wrapper` functionality and provides a `GroupWrapper` to map agent id to some group id.

Vectorized
>>>>>>>>>>

MALib supports interacting with multiple environments in parallel with the implementation of auto-vectorized environment interface implemented in 'malib.rollout.env.vector_env'.

For users who want to use parallel rollout, he/she needs to modify certain contents in `rollout_config`.

.. code-block:: python

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


Advanced Usage / Coming Soon
----------------------------

MALib is now working to introduce more complex scenarios including Google_Research_Football_ and SMAC_.

Also registration for environments similar to OpenAI-Gym is coming soon.

.. _SMAC: https://github.com/oxwhirl/smac
.. _Google_Research_Football: https://github.com/google-research/football