from copyreg import pickle
from re import A
import pytest
import time
import torch
import numpy as np
import os, tempfile, shutil

from pytest_mock import MockerFixture
from pytest import MonkeyPatch

from malib.utils.typing import ParameterDescription, Status
from malib.backend.datapool.parameter_server import ParameterServer, Table

from tests import ServerMixin


class TestTable:
    @pytest.fixture(autouse=True)
    def init(self):
        self.table = Table(
            env_id="test", agent_id="agent_0", policy_id="policy_0", policy_type="test"
        )

    def test_push_parameter(self):
        parameter_desc = ParameterDescription.gen_template(
            env_id=self.table.env_id,
            id=self.table.policy_id,
            data=torch.Tensor([1.0 for _ in range(3)]),
        )
        self.table.insert(parameter_desc)

        # check status
        status = self.table.status
        assert status.locked
        assert status.gradient_status == Status.NORMAL

    @pytest.mark.parametrize("lock", [True, False])
    def test_push_gradients(self, lock):
        # we set a fake parameter here
        parameter_desc = ParameterDescription.gen_template(
            env_id=self.table.env_id,
            id=self.table.policy_id,
            type=ParameterDescription.Type.PARAMETER,
            data=torch.Tensor([1.0 for _ in range(3)]),
            lock=lock,
        )
        self.table.insert(parameter_desc)

        # then one gradient
        assert self.table.parallel_num == 1

        # then 4 pieces of gradients
        self.table.parallel_num = 4
        assert self.table.parallel_num == 4
        gradient_desc = ParameterDescription.gen_template(
            env_id=self.table.env_id,
            id=self.table.policy_id,
            type=ParameterDescription.Type.GRADIENT,
            lock=False,
        )

        for i in range(self.table.parallel_num):
            gradient_desc.data = [np.asarray([0.5 + i for _ in range(3)])]
            self.table.insert(gradient_desc)

        if not lock:
            with pytest.raises(IndexError):
                self.table.gradients = [
                    [np.asarray([0.5 + i for _ in range(3)])]
                    for i in range(self.table.parallel_num)
                ]
                self.table.insert(gradient_desc)

    def test_presistence(self):
        tempdir = tempfile.mkdtemp(prefix="malib-parameter-table-test-")
        try:
            # dump
            file_name = str(time.time())
            self.table.dump(os.path.join(tempdir, file_name))

            # load
            self.table.load(os.path.join(tempdir, f"{file_name}.pkl"))
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(tempdir)


def mock_table_insert(self, desc):
    pass


def mock_table_get(self, desc):
    return desc


@pytest.mark.parametrize(
    "server_config",
    [{}, {"init_job": {}}, {"quit_job": {}}, {"init_job": {}, "quit_job": {}}],
    scope="class",
)
class TestParameterServer(ServerMixin):
    @pytest.fixture(autouse=True)
    def init(self, server_config, mocker: MockerFixture, monkeypatch: MonkeyPatch):
        self.coordinator = self.init_coordinator()
        self.dataset_server = self.init_dataserver()

        monkeypatch.setattr(
            "malib.backend.datapool.parameter_server.Table.insert", mock_table_insert
        )
        monkeypatch.setattr(
            "malib.backend.datapool.parameter_server.Table.get", mock_table_get
        )
        if server_config.get("load_when_start"):
            self.parameter_server.load = mocker.patch.object(
                self.parameter_server, "load"
            )
        self.parameter_server = ParameterServer(exp_cfg=None, **server_config)
        if server_config.get("load_when_start"):
            self.parameter_server.load.assert_called_once()

    @pytest.mark.parametrize(
        "data_type",
        [ParameterDescription.Type.PARAMETER, ParameterDescription.Type.GRADIENT],
        scope="function",
    )
    @pytest.mark.parametrize("keep_return", [False, True], scope="function")
    def test_push_and_pull(self, data_type, keep_return):
        # create a fake table
        desc = ParameterDescription.gen_template(type=data_type, id="policy_0")
        self.parameter_server.push(desc)
        assert self.parameter_server.check_ready(parameter_desc=desc)
        self.parameter_server.pull(desc, keep_return)

    def test_persistence(self):
        desc = ParameterDescription.gen_template(
            type=ParameterDescription.Type.PARAMETER, id="policy_0"
        )
        self.parameter_server.push(desc)
        tempdir = tempfile.mkdtemp(prefix="malib-parameter-server-test-")
        try:
            self.parameter_server.dump(tempdir)
            self.parameter_server.load(file_path=tempdir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(tempdir)

    def test_close(self, mocker: MockerFixture):
        self.parameter_server.dump = mocker.patch.object(self.parameter_server, "dump")
        self.parameter_server.shutdown()
        if self.parameter_server.dump_when_closed:
            self.parameter_server.dump.assert_called_once()
