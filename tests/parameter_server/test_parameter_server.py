import pytest

from pytest_mock import MockerFixture
from pytest import MonkeyPatch

from malib.backend.datapool.parameter_server import ParameterServer

from tests import ServerMixin


class TestTable:
    @pytest.fixture(autouse=True)
    def init(self):
        pass

    def test_push(self):
        pass

    def test_pull(self):
        pass


@pytest.mark.parametrize("data_type,", [("parameter",), ("gradients")])
class TestParameterServer(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(self, server_config):
        self.locals = locals()
        self.coordinator = self.init_coordinator()
        self.dataset_server = self.init_dataserver()
        self.parameter_server = ParameterServer(exp_cfg=None, **server_config)

    def test_pull(self):
        pass

    def test_push(self):
        pass

    def test_dump(self):
        pass

    def test_load(self):
        pass
