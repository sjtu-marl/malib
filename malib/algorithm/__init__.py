from collections import namedtuple

from . import maddpg, dqn, ppo, ddpg, qmix, sac, discrete_sac

from .imitation import bc, advirl


Algorithm = namedtuple("Algorithm", "policy, trainer, loss")
ALGORITHM_LIB = {
    item.NAME: Algorithm(item.POLICY, item.TRAINER, item.LOSS)
    for item in [maddpg, dqn, ppo, ddpg, qmix, sac, discrete_sac, bc]
}

Reward = namedtuple("Reward", "reward, trainer, loss")
REWARD_LIB = {
    item.NAME: Reward(item.REWARD, item.TRAINER, item.LOSS)
    for item in [advirl]
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


def get_reward_algorithm_space(registered_reward_name: str):
    """Return an instance of namedtuple `Reward`. You can retrieve registered reward with following
    instructions.

    Example:
        >>> reward = get_reward_space('ADVIRL')
        >>> policy_cls = reward.reward
        >>> trainer_cls = reward.trainer

    :param registered_reward_name: str, registered reward name
    :return: a named tuple describes the reward and trainer class
    """

    return REWARD_LIB[registered_reward_name]
