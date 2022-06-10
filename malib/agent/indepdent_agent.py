"""
Independent agent interface, for independent algorithms training. Policy/Trainer adding rule: one policy one trainer.
"""

from typing import Dict, Tuple, Any, Callable, List, Union

import shutup

shutup.please()

from malib.utils.typing import AgentID
from malib.utils.tianshou_batch import Batch
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
        self, batch: Union[Batch, Dict[AgentID, Batch]], batch_indices: List[int]
    ) -> Dict[str, Any]:
        if isinstance(batch, Batch):
            return batch
        else:
            raise NotImplementedError
