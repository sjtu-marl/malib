from collections import namedtuple

from . import maddpg, dqn, ppo, ddpg


Algorithm = namedtuple("Algorithm", "policy, trainer, loss")
ALGORITHM_LIB = {
    item.NAME: Algorithm(item.POLICY, item.TRAINER, item.LOSS)
    for item in [maddpg, dqn, ppo, ddpg]
}


def get_algorithm_space(registered_algorithm_name: str):
    """Return an instance of namedtuple `Algorithm`. You can retrieve registered policy with following
    instructions.

    Example:
        >>> algorithm = get_algorithm_space('PPO')
        >>> policy_cls = algorithm.policy
        >>> trainer_cls = algorithm.trainer

    :param registered_algorithm_name: str, registered algorithm name
    :return: a named tuple describes the policy and trainer class
    """

    return ALGORITHM_LIB[registered_algorithm_name]
