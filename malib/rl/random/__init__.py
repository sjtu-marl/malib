from .policy import RandomPolicy
from .random_trainer import RandomTrainer
from .config import Config

Policy = RandomPolicy
Trainer = RandomTrainer
DEFAULT_CONFIG = Config

__all__ = ["Policy", "Trainer", "DEFAULT_CONFIG"]
