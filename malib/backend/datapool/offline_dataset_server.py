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

from typing import Dict, Any

import logging
import ray
import reverb

from malib import settings
from malib.utils.typing import PolicyID, AgentID


logger = logging.getLogger(__name__)


def log(message: str):
    logger.log(settings.LOG_LEVEL, f"(dataset server) {message}")


@ray.remote(num_cpus=0)
class OfflineDataset:
    def __init__(self, table_capacity: int):
        self.servers: Dict[str, reverb.Server] = {}
        self.capacity = table_capacity

    def get_port(self, table_name: str):
        return self.servers[table_name].port

    def get_client_kwargs(self, table_name: str) -> Dict[str, Any]:
        """Retrieve reverb client kwargs.

        Args:
            table_name (str): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        return {"address": self.servers[table_name].port}

    def create_table(self, name: str, reverb_server_kwargs: Dict[str, Any]):
        """Create sub reverb server.

        Args:
            name (str): Sub server name.
            reverb_server_kwargs (Dict[str, Any]): Reverb server kwargs, keys including:
            - `port`: str, optional, the gRPC port string.
            - `checkpointer`: reverb.platform.default.CheckpointerBase, optional.
            - `tb_params_list`: list of dict, required, for constructing tables.

        Note:
            for each item in `tb_params_list`, the dict should include the folloing keys:
            - `name`:
            - `sampler`:
            - `remover`:
            - `max_size`: int, if not given, will use default `self.capacity`.
            - `rate_limiter`:
            - `max_times_sampled`:
            - `extension`:
            - `signature`:
        """

        if name not in self.servers:
            port = reverb_server_kwargs.get("port", None)
            checkpointer = reverb_server_kwargs.get("checkpointer", None)
            tb_params_list = reverb_server_kwargs["tb_params_list"]
            self.servers[name] = reverb.Server(
                tables=[
                    reverb.Table(
                        name=tb_params["name"],
                        sampler=tb_params.get("sampler", reverb.selectors.Uniform()),
                        remover=tb_params.get("remover", reverb.selectors.Fifo()),
                        max_size=tb_params.get("max_size", self.capacity),
                        rate_limiter=tb_params.get(
                            "rate_limiter", reverb.rate_limiters.MinSize(1)
                        ),
                        max_times_sampled=tb_params.get("max_times_sampled", 0),
                        extensions=tb_params.get("extension", ()),
                        signature=tb_params.get("signature", None),
                    )
                    for tb_params in tb_params_list
                ],
                port=port,
                checkpointer=checkpointer,
            )

    def load_from_dataset(
        self,
        file: str,
        env_id: str,
        policy_id: PolicyID,
        agent_id: AgentID,
    ):
        """
        Expect the dataset to be in the form of List[ Dict[str, Any] ]
        """
        raise NotImplementedError
