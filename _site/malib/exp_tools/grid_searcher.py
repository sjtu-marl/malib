from itertools import product
from typing import Union, Dict, Sequence, Any

from malib.exp_tools.tune_type import Grid


class ParameterGrid:
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py
    def __init__(self, param_grid: Union[Dict, Sequence]):
        """Validate data form
        Parameters
        ----------
        param_grid
        """
        if isinstance(param_grid, Dict):
            param_grid = [param_grid]

        if isinstance(param_grid, Sequence):
            for grid in param_grid:
                assert isinstance(grid, Dict)
                for k in grid:
                    assert isinstance(grid[k], Sequence), ""
        else:
            raise ValueError("Should pass a dict or sequence type")

        self._param_grid = param_grid

    def __iter__(self):
        for p in self._param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params


class ConfigHandler:
    @staticmethod
    def get_param_grid(config: Dict[str, Any], *, prefix="") -> Dict:
        param_grid = {}
        for k, v in config.items():
            if isinstance(v, Grid):
                param_grid[prefix + k] = v.data
            elif isinstance(v, Dict):
                param_grid.update(ConfigHandler.get_param_grid(v, prefix=f"{k}_"))
        return param_grid

    @staticmethod
    def build_config(config: Dict[str, Any], cur_param, *, prefix=""):
        res = config.copy()
        for k, v in config.items():
            if isinstance(v, Grid):
                res[k] = cur_param[prefix + k]
            elif isinstance(v, Dict):
                res[k] = ConfigHandler.build_config(v, cur_param, prefix=f"{k}_")
        return res

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._param_grid_iter = iter(
            ParameterGrid(ConfigHandler.get_param_grid(config))
        )

    def __iter__(self):
        for current_param in self._param_grid_iter:
            yield ConfigHandler.build_config(self._config, current_param)

    def next_param(self):
        return next(self._param_grid_iter)


if __name__ == "__main__":
    config = {"a": Grid([0, 1, 2, 3]), "b": {"b1": Grid([0.1, 0.01]), "b2": -1}}

    ch = ConfigHandler(config)
    for cfg in ConfigHandler(config):
        print(cfg)
