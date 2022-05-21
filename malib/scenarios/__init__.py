from importlib import import_module
from pathlib import Path

from malib.scenarios.scenario import Scenario


class _ScenarioCatalog:
    def __init__(self):
        self._catalog = {}


paradigm_catalog = _ScenarioCatalog()


__all__ = ["scenario_catalog"]

# add all modules in this directory to __all__
__all__.extend(
    [
        import_module(f".{f.stem}", __package__)
        for f in Path(__file__).parent.glob("*.py")
        if "__" not in f.stem
    ]
)
