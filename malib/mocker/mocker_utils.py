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

from typing import Sequence, Dict, Any, Callable, List

import time
import ray


from ray.util import ActorPool

from malib.utils.typing import AgentID
from malib.remote.interface import RemoteInterface
from malib.common.strategy_spec import StrategySpec
from malib.rollout.rolloutworker import RolloutWorker


class FakeInferenceClient(RemoteInterface):
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server: ray.ObjectRef,
        max_env_num: int,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: Dict = None,
        training_agent_mapping: Any = None,
        custom_config: Dict[str, Any] = {},
    ):
        self.max_env_num = max_env_num
        self.agents = env_desc["possible_agents"]

    def add_envs(self, maxinum: int) -> int:
        return self.max_env_num

    def close(self):
        time.sleep(0.5)
        return

    def run(
        self,
        agent_interfaces,
        rollout_config,
        dataset_writer_info_dict=None,
    ) -> Dict[str, Any]:
        time.sleep(0.5)
        return {
            "evaluation": [
                {f"agent_reward/{agent}_mean": 1.0 for agent in self.agents}
            ],
            "total_timesteps": 1000,
            "FPS": 100000,
        }


class FakeInferenceServer(RemoteInterface):
    def __init__(
        self,
        agent_id,
        observation_space,
        action_space,
        parameter_server,
        governed_agents,
    ) -> None:
        pass

    def shutdown(self):
        time.sleep(0.5)
        return

    def save(self, model_dir: str):
        print("called save method")
        return

    def compute_action(self, dataframes, runtime_config):
        print("called computation action")
        return None


class FakeRolloutWorker(RolloutWorker):
    def init_agent_interfaces(
        self, env_desc: Dict[str, Any], runtime_ids: Sequence[AgentID]
    ) -> Dict[AgentID, Any]:
        return {}

    def init_actor_pool(
        self,
        env_desc: Dict[str, Any],
        rollout_config: Dict[str, Any],
        agent_mapping_func: Callable,
    ) -> ActorPool:
        return NotImplementedError

    def init_servers(self):
        pass

    def rollout(
        self,
        runtime_strategy_specs: Dict[str, StrategySpec],
        stopping_conditions: Dict[str, Any],
        data_entrypoints: Dict[str, str],
        trainable_agents: List[AgentID] = None,
    ):
        self.set_running(True)
        return {}

    def simulate(self, runtime_strategy_specs: Dict[str, StrategySpec]):
        time.sleep(0.5)
        return {}

    def step_rollout(
        self,
        eval_step: bool,
        rollout_config: Dict[str, Any],
        dataset_writer_info_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pass

    def step_simulation(
        self,
        runtime_strategy_specs_list: Dict[str, StrategySpec],
        rollout_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        pass


from typing import Tuple

from malib.utils.typing import PolicyID
from malib.common.payoff_manager import PayoffManager


class FakePayoffManager(PayoffManager):
    def __init__(
        self,
        agent_names: Sequence[str],
        agent_mapping_func: Callable[[AgentID], str],
        solve_method="fictitious_play",
    ):
        pass

    def expand(self, strategy_specs: Dict[str, StrategySpec]):
        pass

    def get_matchups_eval_needed(
        self, specs_template: Dict[str, StrategySpec]
    ) -> List[Dict[str, StrategySpec]]:
        return [{}]

    def compute_equilibrium(
        self, strategy_specs: Dict[str, StrategySpec]
    ) -> Dict[str, Dict[PolicyID, float]]:
        probs = {}
        for agent, spec in strategy_specs.items():
            probs[agent] = dict(
                zip(spec.policy_ids, [1 / spec.num_policy] * spec.num_policy)
            )
        return probs

    def update_payoff(
        self, eval_data_tups: List[Tuple[Dict[str, StrategySpec], Dict[str, Any]]]
    ):
        pass


from malib.rollout.manager import RolloutWorkerManager


class FakeRolloutManager(RolloutWorkerManager):
    def __init__(
        self,
        experiment_tag: str,
        stopping_conditions: Dict[str, Any],
        num_worker: int,
        agent_mapping_func: Callable,
        rollout_config: Dict[str, Any],
        env_desc: Dict[str, Any],
        log_dir: str,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        self.env_desc = env_desc

    def rollout(self, task_list: List[Dict[str, Any]]) -> None:
        pass

    def wait(self) -> List[Any]:
        time.sleep(0.1)

    def terminate(self):
        pass


from typing import Union, Type
from collections import defaultdict
from malib.agent.manager import TrainingManager


class FakeTrainingManager(TrainingManager):
    def __init__(
        self,
        experiment_tag: str,
        stopping_conditions: Dict[str, Any],
        algorithms: Dict[str, Any],
        env_desc: Dict[str, Any],
        agent_mapping_func: Callable[[AgentID], str],
        training_config: Dict[str, Any],
        log_dir: str,
        remote_mode: bool = True,
        resource_config: Dict[str, Any] = None,
        verbose: bool = True,
    ):
        agent_groups = defaultdict(lambda: set())
        for agent in env_desc["possible_agents"]:
            rid = agent_mapping_func(agent)
            agent_groups[rid].add(agent)

        self.env_desc = env_desc
        self._agent_groups = agent_groups
        self._runtime_ids = tuple(self._agent_groups.keys())
        self._agent_mapping_func = agent_mapping_func
        self.algorithm = algorithms
        self._experiment_tag = experiment_tag

    def add_policies(
        self, interface_ids: Sequence[str] = None, n: Union[int, Dict[str, int]] = 1
    ) -> Dict[str, Type[StrategySpec]]:
        # return a strategy specs that contains n policy
        return {
            agent: StrategySpec(
                identifier=agent,
                policy_ids=[f"policy-{i}" for i in range(n)],
                meta_data={
                    "policy_cls": self.algorithm["default"][0],
                    "kwargs": None,
                    "experiment_tag": self._experiment_tag,
                    "prob_list": [1 / n] * n,
                },
            )
            for agent in self.env_desc["possible_agents"]
        }

    @property
    def runtime_ids(self) -> Tuple[str]:
        return self._runtime_ids

    def run(self, data_request_identifiers: Dict[str, str]):
        pass

    def wait(self) -> List[Any]:
        time.sleep(0.1)

    def cancel_pending_tasks(self):
        pass

    def terminate(self) -> None:
        pass


import contextlib
import ray

from malib.runner import start_servers


@contextlib.contextmanager
def use_ray_env(namespace: str = None):
    """Start a ray cluster and init parameter server and dataset server.

    Yields:
        Tuple[Any, Any]: A tuple of parameter_server and dataset_server.
    """

    parameter_server, dataset_server = None, None
    try:
        if not ray.is_initialized():
            ray.init()
        parameter_server, dataset_server = start_servers()
        yield (parameter_server, dataset_server)
    finally:
        if parameter_server is not None:
            ray.kill(parameter_server)
        if dataset_server is not None:
            ray.kill(dataset_server)
        ray.shutdown()
