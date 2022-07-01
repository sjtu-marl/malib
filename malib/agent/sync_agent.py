"""
Implementation of synchronous agent interface, work with `SyncRolloutWorker`.
"""

from typing import Callable, Dict, Any, Tuple
import gym.spaces
import ray

from malib.utils.typing import AgentID, PolicyID
from malib.agent.indepdent_agent import IndependentAgent


class SyncAgent(IndependentAgent):
    """SyncAgent inherits `IndependentAgent`, but work in synchronous mode. SyncAgent will do optimization after one
    rollout task has been terminated
    """

    def __init__(
        self,
        experiment_tag: str,
        runtime_id: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Tuple[Dict, Dict, Dict]],
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
