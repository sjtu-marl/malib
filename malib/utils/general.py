from typing import OrderedDict
from malib.utils.typing import Dict


def iter_dicts_recursively(d1, d2):
    """Assuming dicts have the exact same structure."""
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]
