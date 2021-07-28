class Error(Exception):
    pass


class RepeatLockingError(Error):
    """ Raised when lock parameter description """

    pass


class UnexpectedType(Error):
    """ Raised when type is not expected """

    pass


class NoEnoughSpace(Error):
    """Raised when population size is not enough for new policies registry"""

    pass


class UnexpectedAlgorithm(Error):
    """Raised when registered an unkown algorithm in agent.AgentInterface"""

    pass


class TypeError(Error):
    """Raised when illegal type"""

    pass


class RepeatedAssignError(Error):
    """Raised when repeated assign value to a not-None dict"""

    pass


class OversampleError(Error):
    """ Raised when over-sample from a offline data table"""

    pass


class NoEnoughDataError(Error):
    pass


class RegisterFailure(Error):
    pass
