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

import ray
import pytest
import numpy as np
from typing import List

from malib.backend.datapool.data_array import NumpyDataArray
from malib.backend.datapool.offline_dataset_server import OfflineDataset, Table, Episode
from malib.utils.typing import BufferDescription


def test_single_external_dataset():
    ray.init(address=None)

    default_columns = [
        Episode.CUR_OBS,
        Episode.ACTION,
        Episode.NEXT_OBS,
        Episode.DONE,
        Episode.REWARD,
        Episode.ACTION_DIST,
    ]

    single_agent_id = "single_agent"
    single_agent_params = {
        "env_id": "single_env",
        "policy_id": "single_policy",
        "capacity": 10000,
        "init_capacity": 10000,
        "other_columns": ["single_other_column1", "single_other_column2"],
    }

    single_agent_data = {
        column_name: np.random.randn(10000, 10, 10)
        for column_name in default_columns + single_agent_params["other_columns"]
    }
    params = single_agent_params.copy()
    params.pop("init_capacity")
    single_agent_episode = Episode(**params)
    single_agent_episode.fill(**single_agent_data)
    print(single_agent_episode.size)
    print(single_agent_episode.capacity)
    table_name = Table.gen_table_name(
        env_id=single_agent_episode.env_id,
        main_id=[single_agent_id],
        pid=[single_agent_episode.policy_id],
    )
    single_agent_table = Table(name=table_name, multi_agent=False)
    single_agent_table.set_episode(
        {single_agent_id: single_agent_episode}, single_agent_episode.capacity
    )
    single_agent_table.fill(**single_agent_data)
    single_agent_table.dump("/tmp/")

    dataset_config = {
        "episode_capacity": 10000,
        # Define the starting job,
        # currently only support data loading
        "init_job": {
            # main dataset loads the data
            "load_when_start": True,
            "path": "/tmp/",
        },
        # Configuration for external resources,
        # possibly multiple dataset shareds with
        # extensive optimization like load balancing,
        # heterogeneously organized data and etc.
        # currently only support Read-Only dataset
        # Datasets and returned data are organized as:
        # ----------------       --------------------
        # | Main Dataset | ---  | External Dataset [i] |
        # ---------------       --------------------
        "extern": {
            "links": [
                {
                    "name": "expert data",
                    "path": "/tmp/",
                    "write": False,
                }
            ],
            "sample_rates": [1],
        },
    }
    exp_config = {"group": "test_offline_dataset_server", "name": "external_dataset"}

    offline_dataset_server = OfflineDataset.options(
        name="OfflineDataset", max_concurrency=1000
    ).remote(dataset_config, exp_config, test_mode=True)

    # major_dataset_keys = list(offline_dataset_server._tables.keys())
    # extern_dataset_keys = list(offline_dataset_server.external_proxy[0]._tables.keys())
    # assert len(major_dataset_keys) == 1
    # assert len(extern_dataset_keys) == 1
    # expected_table_name = Table.gen_table_name(
    #     env_id=single_agent_params["env_id"],
    #     main_id=single_agent_id,
    #     pid=single_agent_params["policy_id"]
    # )
    # assert expected_table_name in major_dataset_keys
    # assert expected_table_name in extern_dataset_keys

    res, info = ray.get(
        offline_dataset_server.sample.remote(
            BufferDescription(
                env_id=single_agent_params["env_id"],
                agent_id=single_agent_id,
                policy_id=single_agent_params["policy_id"],
                batch_size=32,
                sample_mode=None,
            )
        )
    )
    print(info)
    assert isinstance(res.data, List)
    # res.data expected to be
    # List[Dict[ColumnName:str, numpy.ndarray]] of length 2
    assert len(res.data) == 2
    assert list(res.data[0].values())[0].shape == (32, 10, 10)
    assert list(res.data[1].values())[0].shape == (32, 10, 10)
    ray.shutdown()
