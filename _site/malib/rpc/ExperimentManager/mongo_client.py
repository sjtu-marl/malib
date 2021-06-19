import enum
import time
import gridfs
import psutil
import pymongo
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque, defaultdict
from typing import Iterable

from malib.rpc.ExperimentManager.base_client import BaseClient
from malib.utils.configs.config import EXPERIMENT_MANAGER_CONFIG as CONFIG
from malib.utils.typing import Any, Dict, Tuple, Union, EventReportStatus


class DocType(enum.Enum):
    Settings = "SETTINGS"
    HeartBeat = "HEARTBEAT"
    Metric = "METRIC"
    Report = "REPORT"
    Resource = "RESOURCE"
    Payoff = "PAYOFF"
    Logging = "LOGGING"


def extract_dict(target: Dict[str, Any]):
    if target is None:
        return None

    res = {}
    for k, v in target.items():
        if isinstance(v, Dict):
            subdict = extract_dict(v)
            for sk, sv in subdict.items():
                res[f"{k}/{sk}"] = sv
        else:
            res[k] = v
    return res


class MongoClient:
    """
    Some implementation is inspired by SACRED
    (https://github.com/IDSIA/sacred/blob/master/sacred/observers/mongo.py)
    in order to be compatible with omniboard
    Credit and sincere thanks to SCARED :)
    """

    def __init__(
        self,
        host=None,
        port=None,
        log_level=None,
        db_name="expr",
        heart_beat_freq=5,
        heart_beat_max_trail=10,
        nid="",
    ):
        self._server_addr = host
        self._client = pymongo.MongoClient(host=host, port=port)
        self._executor = ThreadPoolExecutor()

        self._config = CONFIG.copy()
        self._config["nid"] = nid
        if log_level:
            self._config["log_level"] = log_level

        self._database = self._client[db_name]
        self._key = None  # current experiment name
        self._expr = None  # current experiment collection
        self._fs = gridfs.GridFS(self._database)

        # background heart beat thread, started after bound to a table
        self._exit_flag = False
        self._hb_thread = threading.Thread(
            target=self._heart_beat,
            args=(heart_beat_freq, heart_beat_max_trail, self._config["nid"]),
        )
        self._hb_thread.daemon = True

    @staticmethod
    def init_expr(
        database: pymongo.database.Database,
        primary: str,
        secondary: str,
        name: str,
        content: Dict[str, Any],
        nid: str,
    ) -> pymongo.database.Database:
        """

        :param database:
        :param primary:
        :param secondary:
        :param name:
        :param content:
        :param nid:
        :return:
        """
        expr_key = f"{primary}-{secondary}"
        collection = database[expr_key]
        expr_doc = collection.count_documents({"type": DocType.Settings.value}, limit=1)

        if expr_doc == 0:
            res = collection.insert_one(
                {"id": nid, "type": DocType.Settings.value, **content}
            )

        return collection

    def __del__(self):
        self._exit_flag = True
        self._hb_thread.join()
        print(f"Mongo logger-{self._config['nid']} destroyed.")

    def _heart_beat(self, freq: int, max_trail: int, nid: str):
        class ProcStatus:
            def __init__(self):
                self.d = {
                    "heartbeat": time.time(),
                    "cpu": 0,
                    "mem": 0,
                    "gpu": None,
                }

        trails_history = deque(maxlen=max_trail)
        current_process = psutil.Process()
        has_gpu = False
        gpu_handlers = []
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_num = pynvml.nvmlDeviceGetCount()
            for gpu_id in range(gpu_num):
                gpu_handlers.append(pynvml.nvmlDeviceGetHandleByIndex(gpu_id))
        except ImportError as e:
            print("MongoClient: can not import pynvml package, gpu monitor disabled")
            has_gpu = False
        except pynvml.NVMLError as e:
            print("MongoClient: can not load nvml lib, gpu monitor disabled")
            has_gpu = False

        while not self._exit_flag:
            status = ProcStatus()
            status.d["cpu"] = current_process.cpu_percent()
            mem_summary = current_process.memory_info()
            status.d["mem"] = (
                mem_summary.rss - mem_summary.shared
                if hasattr(mem_summary, "shared")
                else mem_summary.rss
            )
            gpu_mem_usages = []
            for handler in gpu_handlers:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handler)
                gpu_mem = 0
                for proc in procs:
                    if proc.pid == current_process.pid:
                        gpu_mem += proc.usedGpuMemory
                gpu_mem_usages.append(gpu_mem)
            status.d["gpu"] = gpu_mem_usages

            doc = {
                "id": self._config["nid"],
                "type": DocType.HeartBeat.value,
                **status.d,
            }
            future = self._executor.submit(self._expr.insert_one, doc)
            trails_history.append(future)

            if len(trails_history) > 2:
                trail = trails_history[0]
                if not trail.done():
                    raise RuntimeError(
                        f"Logger{self._config['nid']} heart beat trails waiting overtime"
                    )
                trails_history.popleft()

            time.sleep(freq)

    def _built_in_logging(self, level, content) -> Future:
        if self._config["log_level"] > level:
            return None

        doc = {
            "id": self._config["nid"],
            "type": DocType.Logging.value,
            "level": level,
            "msg": content,
            "time": time.time(),
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def create_table(
        self, primary=None, secondary=None, extra_info: Dict[str, Any] = None
    ):
        _call_params = {
            "database": self._database,
            "primary": primary or self._config["primary"],
            "secondary": secondary or self._config["secondary"],
            "nid": self._config["nid"],
            "name": "Experiment",
            "content": {
                "Primary": primary or self._config["primary"],
                "Secondary": secondary or self._config["secondary"],
                "ExperimentInfo": extract_dict(extra_info),
                "StartTime": time.time(),
                "CreatedBy": self._config["nid"],
            },
        }
        future = self._executor.submit(self.init_expr, **_call_params)
        while not future.done():
            continue

        self._expr = future.result()
        self._hb_thread.start()

    def info(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.INFO, content=content)

    def warning(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.WARNING, content=content)

    def debug(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.DEBUG, content=content)

    def error(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.ERROR, content=content)

    def report(
        self,
        status: EventReportStatus,
        event_id: int,
        metric: Any = None,
        info: str = "",
    ) -> Future:
        """
        Event_id here are mainly used to identify parallel tasks in profiling
        """
        doc = {
            "id": self._config["nid"],
            "type": DocType.Report.value,
            "status": status,
            "event": event_id,
            "info": info,
            "time": time.time(),
        }
        if metric is not None:
            doc.update({"metric": metric})

        return self._executor.submit(self._expr.insert_one, doc)

    def send_scalar(
        self,
        tag=None,
        content=None,
        tag_content_dict: Dict[str, Any] = {},
        global_step=0,
        walltime=None,
        batch_mode: bool = False,
    ) -> Future:
        walltime = walltime or time.time()
        doc_identifier_region = {
            "id": self._config["nid"],
            "type": DocType.Metric.value,
        }

        """
            aggregate = False: parsed as scalar batch, will visualized separately, 
            aggregate = True for send_scalars
        """
        if batch_mode:
            doc_content_region = {
                "aggregate": False,
                "content": [
                    {"name": sub_tag, "content": field_value}
                    for sub_tag, field_value in tag_content_dict.items()
                ],
                "step": global_step,
                "time": walltime,
            }
        else:
            doc_content_region = {
                "aggregate": False,
                "name": tag,
                "content": content,
                "step": global_step,
                "time": walltime,
            }
        return self._executor.submit(
            self._expr.insert_one, {**doc_identifier_region, **doc_content_region}
        )

    def send_scalars(
        self,
        tag: Iterable[str],
        content: Dict[str, Any],
        global_step: int = 0,
        walltime=None,
    ) -> Future:
        """
        Expect the tag_value_dict in the form of
        {str, float} where the strs are taken as tags,
        the floats are taken as values. Or in the form
        of {str, (float, int)}, where ints are taken as
        steps.
        """
        walltime = walltime or time.time()
        doc = {
            "id": self._config["nid"],
            "type": DocType.Metric.value,
            "aggregate": True,
            "name": tag,
            "content": [
                {"name": sub_tag, "content": field_value}
                for sub_tag, field_value in content.items()
            ],
            "step": global_step,
            "time": walltime,
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def send_arbitrary_object(self, f, filename) -> Future:
        file_id = self._fs.put(f, filename=filename)
        doc = {
            "id": self._config["nid"],
            "type": DocType.Resource.value,
            "content": file_id,
            "time": time.time(),
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def get(self, tags: Tuple[str] = None) -> Future:
        _call_params = {"collection": self._expr, "fields": tags}
        return self._executor.submit(self.pull, **_call_params)

    # TODO(jing) How to elegently print payoff matrix?
    def send_obj(
        self, tag, obj, global_step: int = 0, walltime: Union[float, int] = None
    ) -> Future:
        """
        worked as sending payoff matrix
        """
        if tag == "__Payoff__":
            columns = defaultdict(lambda: [])
            for (aid, pid), (raid, reward) in zip(
                obj["Population"].items(), obj["Agents-Reward"]
            ):
                columns["aid"].append(aid)
                columns["pid"].append(pid)
                assert aid == raid
                columns["reward"].append(reward)
            doc = {
                "id": self._config["nid"],
                "type": DocType.Payoff.value,
                "columns": columns,
                "step": global_step,
                "time": walltime or time.time(),
            }
        else:
            raise NotImplementedError
        self._executor.submit(self._expr.insert_one, doc)

    def send_image(self) -> Future:
        raise NotImplementedError
