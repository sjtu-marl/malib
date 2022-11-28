# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Callable, Dict, Any, Type, Tuple, List, Union

from malib.utils.typing import AgentID
from malib.utils.tianshou_batch import Batch
from malib.models.torch import make_net
from malib.agent.agent_interface import AgentInterface


class TeamAgent(AgentInterface):
    def __init__(
        self,
        experiment_tag: str,
        runtime_id: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Tuple[Type, Type, Dict, Dict]],
        agent_mapping_func: Callable[[AgentID], str],
        governed_agents: Tuple[AgentID],
        trainer_config: Dict[str, Any],
        custom_config: Dict[str, Any] = None,
        local_buffer_config: Dict = None,
        verbose: bool = True,
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
            verbose,
        )

        assert (
            "critic" in custom_config
        ), f"TeamAgent must be given a shared critic network"

    def multiagent_post_process(
        self,
        batch_info: Union[
            Dict[AgentID, Tuple[Batch, List[int]]], Tuple[Batch, List[int]]
        ],
    ) -> Dict[str, Batch]:
        assert isinstance(
            batch_info, Dict
        ), "TeamAgent accepts only a dict of batch info"

        res = {}
        for agent in self.governed_agents:
            batch_info[agent][0].to_torch()
            res[agent] = batch_info[agent][0]
        return res
