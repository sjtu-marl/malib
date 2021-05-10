import logging
import wrapt
import os
import inspect
import uuid
from contextlib import contextmanager

from collections import defaultdict

from malib import settings
from malib.rpc.ExperimentManager.ExperimentClient import ExprManagerClient
from malib.rpc.ExperimentManager.mongo_client import MongoClient
from malib.utils.typing import Any, EventReportStatus, MetricType
from malib.utils.aggregators import Aggregator


class Log:
    init = False

    @staticmethod
    @contextmanager
    def stat_feedback(
        log: bool = False, logger: Any = None, worker_idx: str = None, *args, **kwargs
    ):
        """Collect a sequence of statistics (agent wise).

        :param bool log: Send to backend or not.
        :param Any logger: Logger handler, could be MongodbClient or ExprManagerClient.
        :param str worker_idx: Identity of actor, for identification of actor. Default to None
        :param args: List of args.
        :param kwargs: Dict of args
        """

        # __enter__
        statistics = []
        merged_statistics_seq = []
        yield statistics, merged_statistics_seq

        # __exit__
        collected_statistics = defaultdict(
            lambda: defaultdict(
                lambda: {"agg": None, "val": [], "tag": None, "log": False}
            )
        )
        merged_statistics = {}
        scale = len(statistics)
        for statistic in statistics:
            assert isinstance(statistic, dict), statistic.__class__
            for aid, agent_data in statistic.items():
                for key, entry in agent_data.items():
                    tag = entry.tag or f"{aid}/{key}"
                    if collected_statistics[aid][key]["agg"] is None:
                        collected_statistics[aid][key]["agg"] = Aggregator.get(
                            entry.agg
                        )(scale=scale)
                    if collected_statistics[aid][key]["tag"] is None:
                        collected_statistics[aid][key]["tag"] = tag
                    collected_statistics[aid][key]["log"] = log and entry.log
                    collected_statistics[aid][key]["val"].append(entry.value)

        summaries = {}
        for aid, agent_data in collected_statistics.items():
            agent_data_dict = {}
            merged_statistics.update({aid: agent_data_dict})
            for key, entry in agent_data.items():
                val = entry.get("agg").apply(entry.get("val"))
                if entry.get("log"):
                    summaries.update({entry.get("tag"): val})
                # print("Agg: ", entry.get("agg"), " Val: ", val)
                agent_data_dict.update({key: val})

        if log and logger is not None:
            if isinstance(logger, ExprManagerClient):
                for k, v in summaries.items():
                    if worker_idx is not None:
                        k = f"{worker_idx}/{k}"
                    logger.send_scalar(tag=k, content=v, *args, **kwargs)
            elif isinstance(logger, MongoClient):
                logger.send_scalar(
                    tag_content_dict=summaries, batch_mode=True, *args, **kwargs
                )

        merged_statistics_seq.append(merged_statistics)

    @staticmethod
    def data_feedback(enable=True, post_fix=""):
        """Parse raw statistics.

        :param enable: bool, enable logger or not
        :param profiling: bool, enable profiling or not
        :return: a wrapper
        """

        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if not hasattr(instance, "logger") or instance.logger is None or not enable:
                return func(*args, **kwargs)

            if isinstance(instance.logger, MongoClient):
                eid = f"{func.__name__+post_fix}-{uuid.uuid1()}"
                instance.logger.report(
                    status=EventReportStatus.START, event_id=eid, metric=0
                )
            statistics, data = func(*args, **kwargs)
            if isinstance(instance.logger, MongoClient):
                data_entry_count = 0
                byte_count = 0
                for aid, episode in data.items():
                    byte_count += episode.data_bytes
                    data_entry_count += episode.size
                instance.logger.report(
                    status=EventReportStatus.END,
                    event_id=eid,
                    metric=[data_entry_count, byte_count],
                )
            return statistics, data

        return wrapper

    @staticmethod
    @contextmanager
    def timer(
        log: bool = False, logger: Any = None, postfix: str = "", *args, **kwargs
    ):
        if logger is None or not log:
            yield None
        else:
            eid = f"{inspect.stack()[2].function}{postfix}-{uuid.uuid1()}"
            logger.report(status=EventReportStatus.START, event_id=eid, metric=None)
            yield None
            logger.report(
                status=EventReportStatus.END,
                event_id=eid,
                metric=None,
            )

    @staticmethod
    def method_timer(enable=settings.PROFILING):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if (
                instance is None
                or not isinstance(instance.logger, MongoClient)
                or not enable
            ):
                return func(*args, **kwargs)

            eid = f"{func.__name__}-{uuid.uuid1()}"
            instance.logger.report(
                status=EventReportStatus.START, event_id=eid, metric=None
            )
            ret = func(*args, **kwargs)

            instance.logger.report(
                status=EventReportStatus.END,
                event_id=eid,
                metric=None,
            )
            return ret

        return wrapper


def get_logger(
    log_level=logging.INFO,
    log_dir="",
    name=None,
    expr_group=None,
    expr_name=None,
    port="localhost:12333",
    remote=False,
    mongo=False,
    file_stream=True,
    *args,
    **kwargs,
):
    if remote:
        assert expr_group is not None
        assert expr_name is not None

    if remote and mongo:
        try:
            logger = MongoClient(nid=name, log_level=log_level)
            logger.create_table(
                primary=expr_group,
                secondary=expr_name,
                extra_info=kwargs.get("info", None),
            )
        except Exception as e:
            print(e)
        return logger
    elif remote:
        logger = ExprManagerClient(port, nid=name, log_level=log_level)
        create_table_future = logger.create_table(
            primary=expr_group, secondary=expr_name
        )
        while not create_table_future.done():
            continue
        key, recv_time = create_table_future.result()

        logger.info(f"New client start as: key={key} / recv_time={recv_time}")
        return logger
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            return logger
        logger.setLevel(log_level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        if file_stream:
            log_dir = os.path.join(log_dir, "log_files", expr_group, expr_name)
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            except FileExistsError:
                print(f"[WARNING] file exists: {log_dir}")
            file_handler = logging.FileHandler(
                filename=os.path.join(log_dir, f"{name}.log"), mode="w"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
