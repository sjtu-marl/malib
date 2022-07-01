"""
This file gives an implementation of a typical training framework in multi-agent learning: Centralized training and
decentralized execution training workflow. Users can use this agent interface to implement algorithms like MADDPG.
Besides, our implementation can work in a more general case: users can register multiple policies in heterogeneous.

Example:
    >>> algorithm_candidates = {
    ...     "PPO": {},
    ...     "DDPG": {},
    ...     "DQN": {}
    ... }
    >>> learner = CTDEAgent("agent_0", ..., algorithm_candidates)
"""

from typing import Dict, Any, Callable, Tuple, List, Type

import gym

from malib.utils.typing import AgentID, PolicyID
from malib.agent.agent_interface import AgentInterface
from malib.algorithm.common.policy import Policy


class CTDEAgent(AgentInterface):
    """An agent whose learning paradigm is centralized training decentralized execution.
    Notice that, for a single policy in the CTDEAgent, it doesn't need to be CTDE, namely
    In fact, CTDEAgent just specify which global information the inside CTDE-type policy
    can use, e.g. which policies it can pretend to know.
    """

    def __init__(
        self,
        experiment_tag: str,
        runtime_id: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Tuple[Type, Type, Dict]],
        agent_mapping_func: Callable[[AgentID], str],
        governed_agents: Tuple[AgentID],
        trainer_config: Dict[str, Any],
        custom_config: Dict[str, Any] = None,
        local_buffer_config: Dict = None,
    ):
        super().__init__(
            experiment_tag,
            runtime_id,
            log_dir,
            env_desc,
            algorithms,
            agent_mapping_func,
            governed_agents,
            trainer_config,
            custom_config,
            local_buffer_config,
        )
