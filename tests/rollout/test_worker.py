import pytest

from malib.utils.typing import Dict
from malib.rollout.rollout_worker import RolloutWorker

from tests import ServerMixin


class TestRolloutWorker(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(self, env_desc, metric_type, remote: bool, save: bool, kwargs: Dict):
        self.locals = locals()
        self.coordinator = self.init_coordinator()
        self.parameter_server = self.init_parameter_server()
        self.dataset_server = self.init_dataserver()

        self.worker = RolloutWorker(
            "test", env_desc, metric_type, remote, save, **kwargs
        )

    def test_actor_pool_checking(self):
        assert self.locals["remote"]

    def test_simulation_exec(self):
        pass

    def test_rollout_exec(self):
        pass
