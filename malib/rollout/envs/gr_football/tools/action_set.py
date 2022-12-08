import numpy as np

N_ACTIONS = 19

NONE = -1

(
    NO_OP,
    LEFT,
    TOP_LEFT,
    TOP,
    TOP_RIGHT,
    RIGHT,
    BOTTOM_RIGHT,
    BOTTOM,
    BOTTOM_LEFT,
    LONG_PASS,
    HIGH_PASS,
    SHORT_PASS,
    SHOT,
    SPRINT,
    RELEASE_DIRECTION,
    RELEASE_SPRINT,
    SLIDE,
    DRIBBLE,
    RELEASE_DRIBBLE,
) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)


def is_direction(action):
    return action >= LEFT and action <= BOTTOM_LEFT


def is_passing(action):
    return action >= LONG_PASS and action <= SHORT_PASS


DIRECTIONS = list(range(LEFT, LEFT + 8))
PASSINGS = list(range(LONG_PASS, SHORT_PASS + 1))

N_DIRECTIONS = len(DIRECTIONS)

BUILT_IN = 19
