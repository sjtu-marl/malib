"""
Independent agent interface, for independent algorithms training. Policy/Trainer adding rule: one policy one trainer.
"""

from typing import Dict, Tuple, Any, Callable

from malib.utils.typing import AgentID
from malib.agent.agent_interface import AgentInterface


class IndependentAgent(AgentInterface):
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

    def multiagent_post_process(
        self, batch: Dict[AgentID, Dict[str, Any]]
    ) -> Dict[str, Any]:
        return super().multiagent_post_process(batch)
