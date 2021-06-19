# -*- coding: utf-8 -*-

import nashpy as nash
import numpy as np


def compute_two_player_nash(A, B):
    rps = nash.Game(A, B)
    # eqs = rps.support_enumeration()
    # eqs = rps.vertex_enumeration()
    eqs = list(rps.fictitious_play(iterations=20000))[-1]
    return [tuple(map(lambda x: x / np.sum(x), eqs))]
