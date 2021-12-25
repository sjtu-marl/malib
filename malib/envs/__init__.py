from .env import Environment
from .maatari import env_desc_gen as maatari_desc_gen, MAAtari
from .mpe import env_desc_gen as mpe_desc_gen, MPE
from .poker import PokerParallelEnv, env_desc_gen as poker_desc_gen
from .gym import GymEnv, env_desc_gen as gym_desc_gen

# FIXME(ming): check environment installation here
# from .star_craft2 import SC2Env, StatedSC2
# from .smarts.env import SMARTS

_ENV_LIB = {"MAAtari": MAAtari, "MPE": MPE, "Poker": PokerParallelEnv, "Gym": GymEnv}
_ENV_DESC_GEN_LIB = {
    "MAAtari": maatari_desc_gen,
    "MPE": mpe_desc_gen,
    "Poker": poker_desc_gen,
    "Gym": gym_desc_gen,
}


def get_env_cls(name: str):
    return _ENV_LIB[name]


def gen_env_desc(name: str):
    return _ENV_DESC_GEN_LIB[name]
