import numpy as np
from ..tools import action_set
import scipy.spatial as spatial

XSCALE = 54.4
YSCALE = 83.6
ZSCALE = 1

PITCH_OUTTER_WIDTH = 110  # may not use
PITCH_OUTTER_HEIGHT = 72  # may not use
PITCH_OUTTER_X_MAX = PITCH_OUTTER_WIDTH / 2
PITCH_OUTTER_X_MIN = -PITCH_OUTTER_X_MAX
PITCH_OUTTER_Y_MAX = PITCH_OUTTER_HEIGHT / 2
PITCH_OUTTER_Y_MIN = -PITCH_OUTTER_Y_MAX

PITCH_X_MAX = XSCALE * 1
PITCH_X_MIN = -PITCH_X_MAX
PITCH_Y_MAX = YSCALE * 0.42
PITCH_Y_MIN = -PITCH_Y_MAX
PITCH_Z_MAX = 5  # TODO just a guess 5m
PITCH_INNER_X = PITCH_X_MAX * 2  # 108.8
PITCH_INNER_Y = PITCH_Y_MAX * 2  # 70.223...
LINE_HALF_W = 0.06  # ?

GOAL_Y_MAX = YSCALE * 0.044  # 3.67 but actually 3.7 in gamedefines.hpp
GOAL_Y_MIN = -GOAL_Y_MAX
GOAL_Z = 2.5
GOAL_DEPTH = 2.55

PENALTY_AREA_Y_MAX = 20.15 - LINE_HALF_W  # 20.09
PENALTY_AREA_Y_MIN = -PENALTY_AREA_Y_MAX
RIGHT_PENALTY_AREA_X_MIN = PITCH_OUTTER_X_MAX - 16.5 + LINE_HALF_W  # 38.56
LEFT_PENALTY_AREA_X_MAX = -RIGHT_PENALTY_AREA_X_MIN
GOAL_AREA_Y_MAX = 9.8  # just guess
GOAL_AREA_Y_MIN = -GOAL_AREA_Y_MAX
RIGHT_GOAL_AREA_X_MIN = PITCH_OUTTER_X_MAX - 5.7  # just guess
LEFT_GOAL_AREA_X_MAX = -RIGHT_GOAL_AREA_X_MIN

RIGHT_PENALTY_X = PITCH_OUTTER_X_MAX - 11  # 44
LEFT_PENALTY_X = -RIGHT_PENALTY_X

FPS = 10
SPF = 1 / FPS
OBS_X_MAX = 1.0
OBS_X_MIN = -OBS_X_MAX
OBS_Y_MAX = 0.44
OBS_Y_MIN = -OBS_Y_MAX
PITCH_2D_DIST_MAX = np.linalg.norm([PITCH_INNER_X, PITCH_INNER_Y])

# TODO: check this is scaled
BALL_CONTROLLED_DIST = 1.0  # m
BALL_CONTROLLED_HEIGHT = 2.0
BALL_SPEED_VARIATION_THRESH = (
    0.5 * FPS
)  # in general case in each frame speed just change with 0.* if without other force


def tx(x):
    return x * XSCALE


def ty(y):
    return y * YSCALE


def tz(z):
    return z * ZSCALE


def tpos(pos):
    pos = np.array(pos)
    if pos.shape[-1] == 2:
        return pos * np.array([XSCALE, YSCALE])
    else:
        return pos * np.array([XSCALE, YSCALE, ZSCALE])


def normalize_coord(pos):
    pos = np.array(pos)
    if pos.shape[-1] == 2:
        return pos / np.array([PITCH_X_MAX, PITCH_Y_MAX])
    else:
        return pos / np.array([PITCH_X_MAX, PITCH_Y_MAX, PITCH_Z_MAX])


def normalize_dist(dist):
    dist /= PITCH_2D_DIST_MAX
    return dist


# def cal_direction_action(pos1,pos2):
#     '''
#     the direction of pos2 relative to pos1
#     '''
#     # TODO double check this
#     dx=-(pos2[0]-pos1[0])
#     dy=-(pos2[1]-pos1[1])
#     idx=int((np.round(np.arctan2(dy,dx)/np.pi/0.25)+8))%8+1
#     return idx


def get_unsigned_angle(vec1, vec2, degree=True):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    assert (
        vec1.ndim == 1 and vec2.ndim == 1 and len(vec1) == len(vec2)
    ), "batch calculation is not supported now"
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_theta = np.clip(np.dot(vec1 / norm1, vec2 / norm2), -1, 1)
    theta = np.arccos(cos_theta)
    if degree:
        theta = np.rad2deg(theta)
    return theta


def get_dist(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    dpos = pos2 - pos1
    dpos = tpos(dpos)
    dist = np.linalg.norm(dpos, axis=-1)
    return dist


def get_pdist(pos1, pos2):
    pos1 = tpos(pos1)
    pos2 = tpos(pos2)
    dists = spatial.distance.cdist(pos1, pos2)
    return dists


def get_smooth_direction_actions(last_actions, order=2):
    last_action = last_actions[-1]
    if not action_set.is_direction(last_action):
        return action_set.DIRECTIONS
    if (
        order == 2
        and len(last_actions) >= 2
        and action_set.is_direction(last_actions[-2])
    ):
        last2_action = last_actions[-2]
        if (last_action + 2 - last2_action) % 8:
            avail_ids = np.arange(last_action - 1 + 8, last_action + 8 + 2 - 1) % 8 + 1
        elif (last2_action + 2 - last_action) % 8:
            avail_ids = (
                np.arange(last2_action - 1 + 8, last2_action + 8 + 2 - 1) % 8 + 1
            )
        elif (last_action + 1 - last2_action) % 8:
            avail_ids = np.arange(last_action - 1 + 8, last_action + 8 + 1 - 1) % 8 + 1
        elif (last2_action + 1 - last_action) % 8:
            avail_ids = (
                np.arange(last2_action - 1 + 8, last2_action + 8 + 1 - 1) % 8 + 1
            )
        else:
            raise Exception("implementation of smooth direction is wrong!")
    else:
        # the next direction is allow to be 90 degrees away,totally 5 actions
        s = (last_action - 1 + 6) % 8
        avail_ids = np.arange(s, s + 5) % 8 + 1
    return list(avail_ids)


def is_ball_owner(obs):
    return obs["ball_owned_team"] == 0 and obs["active"] == obs["ball_owned_player"]


# def predict_ball_touchable(obs):
#     ball_pos=obs["ball"][0:2]
#     dist2ball=...
#     if obs["ball_owned_team"]==0 and obs["ball_owned_player"]==obs["active"]:
#         return True
#     elif obs["ball_owned_team"]==-1 and


def get_speed(direction):
    coord_speed = get_coord_speed(direction)
    speed = np.linalg.norm(coord_speed, axis=-1)
    return speed


def get_coord_speed(direction):
    direction = np.array(direction)
    direction = tpos(direction) * FPS
    return direction


def out_of_pitch(pos):
    pos = tpos(pos)
    if (
        pos[0] < PITCH_X_MIN
        or pos[0] > PITCH_X_MAX
        or pos[1] < PITCH_Y_MIN
        or pos[1] > PITCH_Y_MAX
    ):
        return True
    return False


def left_goal(pos):
    pos = tpos(pos)
    if (
        pos[1] > GOAL_Y_MIN
        and pos[1] < GOAL_Y_MAX
        and pos[0] > PITCH_X_MAX
        and pos[2] < GOAL_Z
    ):
        return True
    return False


def left_owned_ball(obs):
    return obs["ball_owned_team"] == 0


def right_owned_ball(obs):
    return obs["ball_owned_team"] == 1


def free_ball(obs):
    return obs["ball_owned_team"] == -1


def ball_controlled(obs, team, idx):
    """
    only check spatial relationship.
    """
    assert team in ["left", "right"]
    ball_pos = obs["ball"]
    pos = obs["{}_team".format(team)][idx]
    dist = get_dist(pos, ball_pos[:2])
    ball_z = tz(ball_pos[2])
    if dist < BALL_CONTROLLED_DIST and ball_z < BALL_CONTROLLED_HEIGHT:
        return True
    return False


def ball_pass_event():
    # ball_controlled
    # ball_status changed
    pass


def in_penalty_area(pos, side="right"):
    pos = tpos(pos)
    x, y = pos[:2]
    if side == "right":
        if (
            x > RIGHT_PENALTY_AREA_X_MIN
            and y < PENALTY_AREA_Y_MAX
            and y > PENALTY_AREA_Y_MIN
            and x < PITCH_X_MAX
        ):
            return True
        else:
            return False
    else:
        if (
            x < LEFT_PENALTY_AREA_X_MAX
            and y < PENALTY_AREA_Y_MAX
            and y > PENALTY_AREA_Y_MIN
            and x > PITCH_X_MIN
        ):
            return True
        else:
            return False


def in_goal_area(pos, side="right"):
    pos = tpos(pos)
    x, y = pos[:2]
    if side == "right":
        if (
            x > RIGHT_GOAL_AREA_X_MIN
            and y < GOAL_AREA_Y_MAX
            and y > GOAL_AREA_Y_MIN
            and x < PITCH_X_MAX
        ):
            return True
        else:
            return False
    else:
        if (
            x < LEFT_GOAL_AREA_X_MAX
            and y < GOAL_AREA_Y_MAX
            and y > GOAL_AREA_Y_MIN
            and x > PITCH_X_MIN
        ):
            return True
        else:
            return False


def our_ball_owner_in_enemy_penalty_area(obs):
    return left_owned_ball(obs) and in_penalty_area(
        obs["left_team"][obs["ball_owned_player"]]
    )
