# -*- coding: utf-8 -*-
import os

from collections import defaultdict

import pandas as pd
import numpy as np

from malib.utils.typing import AgentID, Any, Dict

from malib.evaluator.base_evaluator import BaseEvaluator


class SimpleEvaluator(BaseEvaluator):
    class StopMetrics:
        MAX_ITERATION = "max_iteration"

    def __init__(self, **config):
        super().__init__({})

        self._data_record = defaultdict(lambda: defaultdict(list))
        self._look_back = config.get("look_back", 100)

        print("Create a Simple Evaluator with look back = {}".format(self._look_back))

    def evaluate(
        self,
        statistics: Dict[AgentID, Dict[str, Any]],
    ) -> Dict[AgentID, Any]:
        res = defaultdict(dict)
        for aid, result in statistics.items():
            for k, v in result.items():
                self._data_record[aid][k].append(v)
                res[aid][k] = np.mean(self._data_record[aid][k][-self._look_back :])
        return res

    def save_res(self, directory: str):
        for aid in self._data_record:
            df = pd.DataFrame.from_dict(self._data_record[aid])
            if not os.path.exists(directory):
                os.mkdir(directory)
            df.to_csv(f"{directory}/{aid}_result.csv")

    def reset(self):
        self._data_record = defaultdict(lambda: defaultdict(list))
