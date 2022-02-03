import pytest
from malib.algorithm.common.misc import EPSGreedy, OUNoise


def test_ounoise():
    ou_noise = OUNoise(action_dimension=2)
    ou_noise.reset()
    ou_noise.noise()


def test_eps_greedy():
    epsg = EPSGreedy(action_dimension=2)
