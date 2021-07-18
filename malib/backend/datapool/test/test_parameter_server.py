# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/7/16
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----

import os
import ray
import copy
import pytest
import torch
import time

from malib.backend.datapool.parameter_server import (
    Parameter,
    ParameterDescription,
    ParameterDescription,
    ParameterServer,
    PARAMETER_TABLE_NAME_GEN,
)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, out_channels),
        )
        for p in self.layers.parameters():
            torch.nn.init.normal_(p)

    def forward(self, x):
        return self.layers(x)


def test_dump_and_load():
    mlp1 = MLP(in_channels=10, out_channels=20)
    mlp2 = MLP(in_channels=15, out_channels=20)

    x1 = torch.rand(size=(16, 10))
    x2 = torch.rand(size=(16, 15))
    with torch.no_grad():
        y1 = mlp1(x1)
        y2 = mlp2(x2)

    exp_cfg = {"group": "test_parameter_server", "name": "dump_and_load"}

    # dump
    ray.init(address=None)
    parameter_server_config = {
        # configuration for dumping parameters at /tmp/
        "quit_job": {
            "dump_when_closed": True,
            # must ended with slash to indicate it is a directory
            "path": "/tmp/test_ps/",
        }
    }
    parameter_server = ParameterServer.options(
        name="ParameterServer", max_concurrency=1000
    ).remote(test_mode=True, **parameter_server_config, exp_cfg=exp_cfg)
    param_desc1 = ParameterDescription(
        time_stamp=time.time(),
        identify="test_agent_1",
        env_id="test_env",
        id="mlp1",
        type=ParameterDescription.Type.PARAMETER,
        lock=False,
        description={"registered_name": "MLP"},
        data=None,
    )
    param_desc2 = copy.copy(param_desc1)

    param_desc1.data = mlp1.state_dict()
    expected_table_name1 = (
        PARAMETER_TABLE_NAME_GEN(
            env_id=param_desc1.env_id,
            agent_id=param_desc1.identify,
            pid=param_desc1.id,
            policy_type=param_desc1.description["registered_name"],
        )
        + ".pkl"
    )
    status = ray.get(parameter_server.push.remote(param_desc1))
    print(status)

    param_desc2.identify = "test_agent_2"
    param_desc2.id = "mlp2"
    param_desc2.data = mlp2.state_dict()
    expected_table_name2 = (
        PARAMETER_TABLE_NAME_GEN(
            env_id=param_desc2.env_id,
            agent_id=param_desc2.identify,
            pid=param_desc2.id,
            policy_type=param_desc2.description["registered_name"],
        )
        + ".pkl"
    )
    status = ray.get(parameter_server.push.remote(param_desc2))
    print(status)

    # wait for the ps to dump the data
    _ = ray.get(parameter_server.shutdown.remote())
    parameter_server = None
    # check the existence of dumped file
    files = os.listdir(parameter_server_config["quit_job"]["path"])
    assert expected_table_name1 in files
    assert expected_table_name2 in files

    parameter_server_config.update(
        {
            # load the dumped parameters
            "init_job": {
                "load_when_start": True,
                "path": parameter_server_config["quit_job"]["path"],
            },
            # clean the properties of quitting schedule
            "quit_job": {},
        }
    )
    parameter_server = ParameterServer.options(
        name="ParameterServerRec", max_concurrency=1000
    ).remote(test_mode=True, **parameter_server_config, exp_cfg=exp_cfg)

    epsilon = 1e-8

    # clean data
    param_desc1.data = None
    status, mlp1_param = ray.get(
        parameter_server.pull.remote(param_desc1, keep_return=True)
    )
    assert mlp1_param.data
    mlp1.load_state_dict(mlp1_param.data)
    with torch.no_grad():
        y1_rec = mlp1(x1)
    res = torch.sub(y1, y1_rec)
    assert torch.all(res < epsilon).item()

    param_desc2.data = None
    status, mlp2_param = ray.get(
        parameter_server.pull.remote(param_desc2, keep_return=True)
    )
    mlp2.load_state_dict(mlp2_param.data)
    with torch.no_grad():
        y2_rec = mlp2(x2)
    res = torch.sub(y2, y2_rec)
    assert torch.all(res < epsilon).item()

    _ = ray.get(parameter_server.shutdown.remote())
    ray.shutdown()
