import pytest

from malib.utils.typing import Dict, Any, RolloutDescription, TaskDescription, TaskType
from malib.rollout.rollout_worker import RolloutWorker

from tests import ServerMixin


@pytest.mark.parametrize(
    "env_desc,metric_type,kwargs",
    [
        (
            None,
            None,
            {
                "num_rollout_actors": 2,
                "num_eval_actors": 1,
                "exp_cfg": None,
                "use_subproc_env": False,
                "batch_mode": False,
                "postprocessor_types": ["default"],
            },
        )
    ],
)
class TestRolloutWorker(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(self, env_desc: Dict[str, Any], metric_type: str, kwargs: Dict):
        self.locals = locals()
        self.coordinator = self.init_coordinator()
        self.parameter_server = self.init_parameter_server()
        self.dataset_server = self.init_dataserver()

        # TODO(ming): mock agent interface here (before initialization)
        self.worker = RolloutWorker("test", env_desc, metric_type, **kwargs)

    def test_actor_pool_checking(self):
        num_eval_actors = self.locals["kwargs"]["num_eval_actors"]
        num_rollout_actors = self.locals["kwargs"]["num_rollout_actors"]

        assert len(self.worer.actors) == num_eval_actors + num_rollout_actors

    def test_sample_dist_config(self):
        # three kinds of task descriptions: 1) rollout; 2) simulation and 3) poilcy add.
        task_desc = TaskDescription(
            task_type=TaskType.SIMULATION,
            content=RolloutDescription(
                agent_involve_info=None,
                fragment_length=None,
                num_episodes=None,
                episode_seg=None,
                terminate_mode=None,
            ),
            state_id="test",
        )
        self.worker.set_state(task_desc)

    def test_simulation_exec(self):
        # TODO(ming): mock stepping here (before execution)
        pass

    def test_rollout_exec(self):
        # TODO(ming): mock stepping here (before execution)
        pass
