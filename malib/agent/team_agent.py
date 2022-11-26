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

from typing import Callable, Dict, Any, Type, Tuple, List

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

        critic_cls = custom_config["critic"]["cls"]
        critic_kwargs = custom_config["critic"]["kwargs"]

        # TODO(ming): get device
        observation_space = None
        action_space = None

        self.team_critic = make_net(observation_space, action_space, device=device)

    def multiagent_post_process(
        self, batch: Dict[AgentID, Batch], batch_indices: List[int]
    ) -> Dict[str, Any]:
        return super().multiagent_post_process(batch, batch_indices)
