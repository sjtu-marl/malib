import abc
import os

import yaml
from typing import Dict, Any, Callable


class BaseConfigFormatter(abc.ABC):
    @staticmethod
    def parse():
        raise NotImplementedError


class DefaultConfigFormatter(BaseConfigFormatter):
    pattern_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "default.yaml"
    )

    @staticmethod
    def parse(global_configs):
        records = None
        print(
            f"Loading parsing config yaml file from {DefaultConfigFormatter.pattern_file_path}"
        )
        with open(DefaultConfigFormatter.pattern_file_path, "r") as f:
            records = yaml.load(f, Loader=yaml.SafeLoader)
        return DefaultConfigFormatter.fill(records, global_configs)

    @staticmethod
    def fill(dst, src) -> Dict[str, Any]:
        if dst is not None and src is not None:
            for k, v in dst.items():
                if v is not None:
                    DefaultConfigFormatter.fill(v, src.get(k, None))
                else:
                    val = src.get(k, None)
                    if isinstance(val, Callable):
                        val = val.__name__
                    dst.update({k: val})
        return dst
