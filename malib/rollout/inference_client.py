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

from argparse import Namespace
from typing import Type, Union, Any, List, Dict, Tuple
from types import LambdaType
from collections import defaultdict

import os
import gc

import ray
import reverb

from ray.util.queue import Queue
from reverb.client import Writer as ReverbWriter

from malib.remote.interface import RemoteInterFace
from malib.utils.typing import (
    AgentID,
    DataFrame,
    BehaviorMode,
    EnvID,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.episode import Episode, Episode, NewEpisodeDict
from malib.utils.preprocessor import get_preprocessor
from malib.utils.timing import Timing
from malib.envs.vector_env import VectorEnv
from malib.envs.async_vector_env import AsyncVectorEnv, AsyncSubProcVecEnv
from malib.rollout.postprocessor import get_postprocessor
from malib.rollout.inference_server import InferenceWorkerSet


def wait_recv(recv_queue: Dict[str, Queue]):
    while True:
        ready = True
        for recv in recv_queue.values():
            if recv.empty():
                ready = False
        if ready:
            break
        # else:
        #     time.sleep(1)


def recieve(queue: Dict[str, Queue]) -> Dict[AgentID, DataFrame]:
    """Recieving messages from remote server.

    :param queue: A dict of queue.
    :type queue: Dict[str, Queue]
    """

    wait_recv(queue)

    rets = {}
    for runtime_id, v in queue.items():
        rets[runtime_id] = v.get_nowait()
    return rets


def postprocessing(episodes, postprocessor_types, policies=None):
    postprocessor_types = ["default"]
    for handler in get_postprocessor(postprocessor_types):
        episodes = handler(episodes, policies)
    return episodes


class InferenceClient(RemoteInterFace):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: LambdaType = None,
    ):
        """Construct an inference client.

        Args:
            env_desc (Dict[str, Any]): Environment description
            dataset_server (_type_): A ray object reference.
            max_env_num (int): The maximum of created environment instance.
            use_subproc_env (bool, optional): Indicate subproc envrionment enabled or not. Defaults to False.
            batch_mode (str, optional): Batch mode, could be `time_step` or `episode` mode. Defaults to "time_step".
            postprocessor_types (Dict, optional): Post processor type list. Defaults to None.
            training_agent_mapping (LambdaType, optional): Agent mapping function. Defaults to None.
        """

        self.dataset_server = dataset_server
        self.use_subproc_env = use_subproc_env
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types or ["defaults"]
        self.process_id = os.getpid()
        self.timer = Timing()
        self.training_agent_mapping = training_agent_mapping or (lambda agent: agent)
        self.max_env_num = max_env_num

        agent_group = defaultdict(lambda: [])
        runtime_agent_ids = []
        for agent in env_desc["possible_agents"]:
            runtime_id = training_agent_mapping(agent)
            agent_group[runtime_id].append(agent)
            runtime_agent_ids.append(runtime_id)
        self.runtime_agent_ids = set(runtime_agent_ids)
        self.agent_group = dict(agent_group)

        obs_spaces = env_desc["observation_spaces"]
        act_spaces = env_desc["action_spaces"]
        env_cls = env_desc["creator"]
        env_config = env_desc["config"]

        self.preprocessor = {
            agent: get_preprocessor(obs_spaces[agent])(obs_spaces[agent])
            for agent in env_desc["possible_agents"]
        }

        if use_subproc_env:
            self.env = AsyncSubProcVecEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )
        else:
            self.env = AsyncVectorEnv(
                obs_spaces, act_spaces, env_cls, env_config, preset_num_envs=max_env_num
            )

        self.recv_queue = None
        self.send_queue = None
        self.reverb_clients: Dict[str, Type[reverb.Client]] = {}

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these \
            new environment instances into it,otherwise do nothing.

        Args:
            maximum (int): Maximum limits.

        Returns:
            int: The number of nested environments.
        """

        if not isinstance(self.env, VectorEnv):
            return 1

        existing_env_num = getattr(self.env, "num_envs", 1)

        if existing_env_num >= maximum:
            return self.env.num_envs

        self.env.add_envs(num=maximum - existing_env_num)

        return self.env.num_envs

    def close(self):
        if self.recv_queue is not None:
            _ = [e.shutdown(force=True) for e in self.recv_queue.values()]
            _ = [e.shutdown(force=True) for e in self.send_queue.values()]
        self.env.close()

    def run(
        self,
        agent_interfaces: Dict[AgentID, InferenceWorkerSet],
        desc: Dict[str, Any],
        dataserver_entrypoint: str = None,
        reset: bool = False,
    ) -> Union[List, Dict]:

        # reset timer, ready for monitor
        self.timer.clear()
        task_type = desc["flag"]

        server_runtime_config = desc.copy()
        server_runtime_config.update(
            {
                "sample_mode": "once",
                # TODO(ming): move to policy
                "preprocessor": self.preprocessor,
            }
        )
        request = Namespace(**desc)

        if task_type == "rollout":
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLORATION
        elif task_type in ["evaluation", "simulation"]:
            server_runtime_config["behavior_mode"] = BehaviorMode.EXPLOITATION

        if self.recv_queue is None or reset:
            self.recv_queue = {
                runtime_id: Queue(actor_options={"num_cpus": 0})
                for runtime_id in agent_interfaces
            }
            self.send_queue = {
                runtime_id: Queue(actor_options={"num_cpus": 0})
                for runtime_id in agent_interfaces
            }

        with self.timer.timeit("inference_server_connect"):
            _ = ray.get(
                [
                    server.connect.remote(
                        [self.recv_queue[runtime_id], self.send_queue[runtime_id]],
                        runtime_config=server_runtime_config,
                        runtime_id=self.process_id,
                    )
                    for runtime_id, server in agent_interfaces.items()
                ],
                timeout=10.0,
            )
            gc.collect()

        if dataserver_entrypoint is not None:
            if dataserver_entrypoint not in self.reverb_clients:
                with self.timer.timeit("dataset_sever_connect"):
                    address = ray.get(
                        self.dataset_server.get_client_kwargs.remote(
                            dataserver_entrypoint
                        )
                    )["address"]
                    self.reverb_clients[dataserver_entrypoint] = reverb.Client(address)
            reverb_writer: ReverbWriter = self.reverb_clients[
                dataserver_entrypoint
            ].writer(max_sequence_length=desc["fragment_length"])

        else:
            reverb_writer: ReverbWriter = None

        results = self.env_runner(collect_backend=reverb_writer)
        return res
