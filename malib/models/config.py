from typing import Type, Dict, Any

from dataclasses import dataclass


@dataclass
class ModelConfig:

    model_cls: Type

    model_args: Dict[str, Any]

    def to_dict(self):
        _dict = self.__dict__.copy()
        return _dict
