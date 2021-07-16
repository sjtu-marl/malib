from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Sequence, List, Dict, Any

from operator import mul
from functools import reduce

from malib import settings
from malib.utils.typing import AgentID, MetricType, PolicyID, MetricEntry
from malib.utils.aggregators import Aggregator


def to_metric_entry(data: Dict[str, Any], prefix=""):
    """Convert a dict of metrics to a dict or metric entries.

    :param Dict[str,Any] data: Raw metrics dict.
    :return: A dict of metric entries.
    """

    res: Dict[str, MetricEntry] = {}
    for k, v in data.items():
        if isinstance(v, MetricEntry):
            res[k] = v
        else:
            res[k] = MetricEntry(
                value=v,
                agg="mean",
                tag=f"{prefix}/{k}",
                log=settings.STATISTIC_FEEDBACK,
            )
    return res


class Metric(metaclass=ABCMeta):
    def __init__(self, agents: List[AgentID]):
        self._agents = agents
        self._episode_data = dict()
        self._statistics = dict()
        # single for sequential, vector for simultaneous
        self._mode = "single"

    @property
    def step_mode(self) -> str:
        return self._mode

    @abstractmethod
    def step(self, agent_id, policy_id, **kwargs) -> None:
        """Record sampled data and save to do evaluation."""

    @abstractmethod
    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Parse episode data and filter with given keys (agent level)"""

    @staticmethod
    def merge_parsed(
        agent_result_seq: Sequence[Dict[AgentID, Any]]
    ) -> Dict[AgentID, Dict[str, float]]:
        """Merge multiple evaluated results."""

    def reset(self, mode="single"):
        self._mode = mode
        self._episode_data = dict()
        self._statistics = dict()


class SimpleMetrics(Metric):
    def __init__(self, agents: List[AgentID]):
        super(SimpleMetrics, self).__init__(agents)
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(
            lambda: defaultdict(
                lambda: MetricEntry(value=0, agg="mean", tag="", log=False)
            )
        )
        self._pids = {}

    def step(self, agent_id, policy_id, **kwargs) -> None:
        self._episode_data[MetricType.REWARD][agent_id].append(kwargs["reward"])
        self._pids[agent_id] = policy_id

    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        for item_key, agent_data in self._episode_data.items():
            # if filter is not None use filter else agents
            for aid in agent_filter or self._agents:
                if item_key == MetricType.REWARD:
                    if self.step_mode == "vector":
                        agent_data[aid] = [
                            sum(e) / max(1, len(e)) for e in agent_data[aid]
                        ]
                    self._statistics[aid][MetricType.REWARD] = MetricEntry(
                        value=sum(agent_data[aid]),
                        agg=Aggregator.MEAN,
                        tag=f"{self._pids[aid]}/{MetricType.REWARD}"
                        if self._pids.get(aid) is not None
                        else MetricType.REWARD,
                        log=True,
                    )
        return self._statistics

    @staticmethod
    def merge_parsed(
        agent_result_seq: Sequence,
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Aggregates a sequence of evaluated results in average, and return an agent dict."""

        agent_res = {}

        for agent_result in agent_result_seq:
            for agent_id, result in agent_result.items():
                if agent_res.get(agent_id, None) is None:
                    tmp = result[MetricType.REWARD]
                    agent_res[agent_id] = {
                        MetricType.REWARD: MetricEntry(
                            value=tmp.value / len(agent_result_seq),
                            agg=tmp.agg,
                            tag=tmp.tag,
                            log=tmp.log,
                        )
                    }
                agent_res[agent_id][MetricType.REWARD].value += result[
                    MetricType.REWARD
                ].value / len(agent_result_seq)

        return agent_res

    def reset(self, mode: str = "single"):
        self._mode = mode
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(
            lambda: defaultdict(
                lambda: MetricEntry(value=0, agg="mean", tag="", log=False)
            )
        )


class JointDistMetric(Metric):
    class Meta:
        REWARD = MetricType.REWARD
        ACTION_DIST = "action_dist"

    def __init__(self, agents: List[AgentID]):
        # must be list here
        agents = list(agents)
        super(JointDistMetric, self).__init__(agents)
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
            "action_dist": defaultdict(lambda: []),
        }
        self._statistics = defaultdict(
            lambda: defaultdict(
                lambda: MetricEntry(value=0, agg="mean", tag="", log=False)
            )
        )
        self._pids = {}

    def step(self, agent_id, policy_id, **kwargs):
        self._episode_data[self.Meta.REWARD][agent_id].append(kwargs[MetricType.REWARD])
        self._episode_data[self.Meta.ACTION_DIST][agent_id].append(
            kwargs[self.Meta.ACTION_DIST]
        )
        self._pids[agent_id] = policy_id

    def _cum_reward_on_joint_dist(self, main, others):
        """Calculate cumulative reward using joint policy distribution"""

        rewards = self._episode_data[MetricType.REWARD][main]
        all_dist = self._episode_data[self.Meta.ACTION_DIST]
        main_dist = [0.0 for _ in range(len(all_dist[main]))]

        if len(others):
            for i, _ in enumerate(main_dist):
                main_dist[i] = reduce(mul, [1.0] + [all_dist[o][i] for o in others])
        else:
            # return all ones
            main_dist = [1.0] * len(main_dist)

        # the head reward from sequential mode is no use
        total_reward = sum(
            [r * dist for dist, r in zip(main_dist, rewards[-len(main_dist) :])]
        )
        return total_reward

    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Parse episode data, return an agent wise MetricEntry dictionary"""

        # if filter is not None use filter else agents
        for i, aid in enumerate(agent_filter or self._agents):
            others = self._agents[:i] + self._agents[i + 1 :]
            if self._pids.get(aid) is not None:
                prefix = f"{aid}/{self._pids[aid]}"
            else:
                prefix = f"{aid}"
            self._statistics[aid][MetricType.REWARD] = MetricEntry(
                value=self._cum_reward_on_joint_dist(aid, others),
                agg="mean",
                tag=f"{prefix}/{MetricType.REWARD}",
                log=True,
            )
        return self._statistics

    @staticmethod
    def merge_parsed(
        agent_result_seq: Sequence[Dict[AgentID, Any]]
    ) -> Dict[AgentID, Dict[str, float]]:
        agent_res = {}
        for agent_result in agent_result_seq:
            for agent_id, result in agent_result.items():
                if agent_res.get(agent_id, None) is None:
                    agent_res[agent_id] = {
                        MetricType.REWARD: 0,
                    }
                if isinstance(result[MetricType.REWARD], MetricEntry):
                    e = result[MetricType.REWARD].value
                else:
                    e = result[MetricType.REWARD]
                agent_res[agent_id][MetricType.REWARD] += e / len(agent_result_seq)

        return agent_res

    def reset(self):
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(lambda: {MetricType.REWARD: 0.0})


METRIC_TYPES = {"simple": SimpleMetrics, "jointdist": JointDistMetric}


def get_metric(metric_type: str):
    """Return a metric handler with given name.

    :param str metric_type: Registered metric type.
    """
    return METRIC_TYPES[metric_type]
