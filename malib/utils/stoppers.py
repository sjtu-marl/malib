from malib.utils.typing import Status, MetricType, MetricEntry, Dict, List


class Stopper:
    def __init__(self, config: Dict, tasks: List = None):
        """Create a stopper instance with metric fields. This fields should cover
        all feasible attributes from rollout/training results.

        :param Dict config: Configuration to control the stopping.
        :param List tasks: A list of sub task identifications. Default to None
        """
        self._config = config
        self.tasks_status = dict.fromkeys(tasks, Status.NORMAL) if tasks else {}

    def __call__(self, results: Dict[str, MetricEntry], global_step: int) -> bool:
        """Parse results and determine whether we should terminate tasks."""

        raise NotImplementedError

    @property
    def info(self):
        """Return statistics for analysis"""

        raise NotImplementedError

    def set_terminate(self, task_id: str) -> None:
        """Terminate sub task tagged with task_id, and set status to terminate."""

        assert task_id in self.tasks_status, (task_id, self.tasks_status)
        self.tasks_status[task_id] = Status.TERMINATE

    def all(self):
        """Judge whether all tasks have been terminated

        :return: a bool value indicates terminated or not
        """

        terminate = len(self.tasks_status) > 0
        for status in self.tasks_status.values():
            if status == Status.NORMAL:
                terminate = False
                break

        return terminate


class SimpleRolloutStopper(Stopper):
    """SimpleRolloutStopper will check the equivalence between evaluate results and"""

    def __init__(self, config, tasks: List = None):
        super(SimpleRolloutStopper, self).__init__(config, tasks)
        self._config["max_step"] = self._config.get("max_step", 100)
        self._info = {MetricType.REACH_MAX_STEP: False}

    @property
    def max_iteration(self):
        return self._config["max_step"]

    def __call__(self, results: Dict[str, MetricEntry], global_step):
        """Default rollout stopper will return true when global_step reaches to an oracle"""
        if global_step == self._config["max_step"]:
            self._info[MetricType.REACH_MAX_STEP] = True
            return True
        return False

    @property
    def info(self):
        raise self._info


class NonStopper(Stopper):
    """NonStopper always return false"""

    def __init__(self, config, tasks=None):
        super(NonStopper, self).__init__(config, tasks)

    def __call__(self, *args, **kwargs):
        return False

    @property
    def info(self):
        return {}


class SimpleTrainingStopper(Stopper):
    """SimpleRolloutStopper will check the equivalence between evaluate results and"""

    def __init__(self, config: Dict, tasks: List = None):
        super(SimpleTrainingStopper, self).__init__(config, tasks)
        self._config["max_step"] = self._config.get("max_step", 100)
        self._info = {
            MetricType.REACH_MAX_STEP: False,
        }

    def __call__(self, results: Dict[str, MetricEntry], global_step):
        """Ignore training loss, use global step."""

        if global_step == self._config["max_step"]:
            self._info[MetricType.REACH_MAX_STEP] = True
            return True
        return False

    @property
    def info(self):
        return self._info


def get_stopper(name: str):
    """Return a stopper class with given type name.

    :param str name: Stopper name, choices {simple_rollout, simple_training}.
    :return: A stopper type.
    """

    return {
        "none": NonStopper,
        "simple_rollout": SimpleRolloutStopper,
        "simple_training": SimpleTrainingStopper,
    }[name]
