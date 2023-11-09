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

from typing import Dict, Tuple, Any, Callable, List, Type, Union

import gym

from gym import spaces
from malib.backend.dataset_server.data_loader import DynamicDataset

from malib.utils.typing import AgentID
from malib.utils.tianshou_batch import Batch
from malib.learner.learner import Learner


class IndependentAgent(Learner):
    def multiagent_post_process(
        self,
        batch_info: Union[
            Dict[AgentID, Tuple[Batch, List[int]]], Tuple[Batch, List[int]]
        ],
    ) -> Dict[str, Any]:
        if not isinstance(batch_info, Tuple):
            raise TypeError(
                "IndependentAgent support only a tuple of batch info as input."
            )

        batch = batch_info[0]
        batch.to_torch(device=self.device)

        return batch
