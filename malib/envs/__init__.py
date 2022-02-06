from .env import Environment
from .maatari import env_desc_gen as maatari_desc_gen, MAAtari
from .mpe import env_desc_gen as mpe_desc_gen, MPE
from .poker import PokerParallelEnv, env_desc_gen as poker_desc_gen
from .gym import GymEnv, env_desc_gen as gym_desc_gen

# FIXME(ming): check environment installation here
from .star_craft2 import SC2Env

# from .smarts.env import SMARTS


_ENV_DESC_GEN_LIB = {
    "MAAtari": maatari_desc_gen,
    "MPE": mpe_desc_gen,
    "Poker": poker_desc_gen,
    "Gym": gym_desc_gen,
}


def get_env_cls(name: str):
    if name == "MAAtari":
        return MAAtari

    if name == "MPE":
        return MPE

    if name == "Poker":
        return PokerParallelEnv

    if name == "Gym":
        return GymEnv

    if name == "SC2":
        return SC2Env

    return ValueError("Unregistered environment: {}".format(name))


def gen_env_desc(name: str):
    if name == "MAAtari":
        return maatari_desc_gen

    if name == "MPE":
        return mpe_desc_gen

    if name == "Poker":
        return poker_desc_gen

    if name == "Gym":
        return gym_desc_gen

    if name == "SC2":
        return
