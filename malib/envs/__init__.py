from .env import Environment
from .maatari.env import MAAtari
from .mpe.env import MPE
from .poker import PokerParallelEnv
from .gym.env import GymEnv

# from .star_craft2 import SC2Env
# from .smarts.env import SMARTS

_LIB = {"MAAtari": MAAtari, "MPE": MPE, "Poker": PokerParallelEnv, "Gym": GymEnv}


def get_env_cls(name: str):
    return _LIB[name]
