import os
import pymongo
import time
import pprint
import http.server
import socketserver

import numpy as np
import threading
from collections import defaultdict

from malib.rpc.ExperimentManager.mongo_client import DocType


class ProfilingHtmlHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.path = "profiler.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


class MongoProfiler:
    exit_flag = False

    def __init__(self, db_host=None, db_port=None, db_name="expr", expr_name=""):
        self._conn = pymongo.MongoClient(host=db_host, port=db_port)
        self._db = self._conn[db_name]
        self._expr = self._db[expr_name]
        self._start_time = self._expr.find_one(
            filter={"type": DocType.Settings.value}, projection={"StartTime": 1}
        ).get("StartTime")
        self._resource_projection_list = ["id", "heartbeat", "cpu", "mem", "gpu"]
        self._throughput_projection_list = ["id", "status", "event", "metric", "time"]
        self._metric_projection_list = [
            "id",
            "name",
            "content",
            "step",
            "time",
            "aggregate",
        ]

        # self._t = threading.Thread(target=self.profiling, args=(True, 4), )
        # self._t.daemon = True

    def get_expr_meta_info(self):
        return self._expr.find(filter={"type": DocType.Settings.value})

    def resource_monitoring(self, project=True):
        records = defaultdict(lambda: defaultdict(lambda: []))

        for client_record in self._expr.find(
            filter={"type": DocType.HeartBeat.value},
            projection={field: 1 for field in self._resource_projection_list}
            if project
            else None,
        ):
            client_data_summary = records[client_record.get("id")]
            for appending_field in self._resource_projection_list[1:]:
                client_data_summary[appending_field].append(
                    client_record.get(appending_field)
                )

        for cid, summary in records.items():
            summary["time"] = np.round(
                np.array(summary["heartbeat"]) - self._start_time, 1
            )
            summary["aliveness"] = summary["heartbeat"][-1] - time.time()
            summary["cpu"] = np.round(np.array(summary["cpu"]), 1)
            summary["mem"] = np.round(np.array(summary["mem"]) / (1024 ** 3), 1)
            summary["gpu"] = np.round(np.array(summary["gpu"]) / (1024 ** 3), 1)

        return records

    def throughput_monitoring(self, project=True):
        records = defaultdict(lambda: [])

        for event in self._expr.find(
            filter={"type": DocType.Report.value, "metric": {"$exists": True}},
            projection={field: 1 for field in self._throughput_projection_list}
            if project
            else None,
        ):
            records[event.get("id")].append(event)

        for cid, events in records.items():
            emap = {}
            aggregated_events = []
            for event in events:
                eid = event.get("event")
                rec = emap.pop(eid, None)
                if rec is not None:
                    assert rec.get("status") == "start"
                    duration = event.get("time") - rec.get("time")
                    event.update({"duration": duration})
                    aggregated_events.append(event)
                else:
                    emap.update({eid: event})
            aggregated_events.sort(key=lambda x: x.get("time"))

            summary = {}
            loaded_time = 0
            timestamps = []
            entry_count = 0
            entry_counts = []
            entry_size = 0
            entry_sizes = []
            for event in aggregated_events:
                loaded_time += event.get("duration")
                timestamps.append(event.get("time"))
                # entry_count += event.get("metric")[0]
                # entry_size += event.get("metric")[1]
                # entry_counts.append(entry_count)
                # entry_sizes.append(entry_size)
                entry_counts.append(event.get("metric")[0])
                entry_sizes.append(event.get("metric")[1])

            summary["name"] = cid
            summary["loaded"] = loaded_time / timestamps[-1]
            summary["time"] = np.round(np.array(timestamps) - self._start_time, 1)
            summary["data entry"] = np.round(np.array(entry_counts) / 1000, 1)
            summary["exchange scale"] = np.round(np.array(entry_sizes) / (1024 ** 2), 1)

            records[cid] = summary

        return records

    def metric_monitoring(self, project=True):
        records = defaultdict(lambda: defaultdict(lambda: []))
        summaries = {}
        for client_record in self._expr.find(
            filter={"type": DocType.Metric.value},
            projection={field: 1 for field in self._metric_projection_list}
            if project
            else None,
        ):
            global_step = client_record.get("step")
            walltime = client_record.get("time") - self._start_time
            content = client_record.get("content")
            if isinstance(content, list):
                if client_record.get("aggregate"):
                    name = client_record.get("name")
                    summary_data = records[name]
                    for kv_pair in content:
                        k = kv_pair["name"]
                        v = kv_pair["content"]
                        summary_data[k].append((v, global_step, walltime))
                else:
                    for kv_pair in content:
                        k = kv_pair["name"]
                        v = kv_pair["content"]
                        records[k]["default"].append((v, global_step, walltime))
            else:
                name = client_record.get("name")
                v = client_record.get("content")
                summary_data = records[name]
                summary_data["default"].append((v, global_step, walltime))

        for graph_name, graph_summary in records.items():
            summary = {}
            for plot_name, plot_points in graph_summary.items():
                values = []
                steps = []
                times = []
                for (v, s, t) in plot_points:
                    values.append(v)
                    steps.append(s)
                    times.append(t)
                values = np.round(values, 3).tolist()
                times = np.round(times, 1).tolist()
                summary.update({plot_name: (times, steps, values)})
            summaries.update({graph_name: summary})
        return summaries

    def event_monitoring(self, project=True):
        records = []
        import re

        type_pattern = re.compile(r"([_a-zA-Z]*)-[0-9]*")
        for event in self._expr.find(
            filter={"type": DocType.Report.value},
            projection={field: 1 for field in self._throughput_projection_list}
            if project
            else None,
        ):
            records.append(event)
        records.sort(key=lambda x: x.get("time"))

        final_records = []
        for event in records:
            converted_event = {
                "name": event.get("event"),
                "cat": re.search(type_pattern, event.get("event")).group(1),
                "pid": event.get("id"),
                "ph": "B" if event.get("status") == "start" else "E",
                "ts": event.get("time") * 1e6,
                "args": {"num_entry/size(bytes)": event.get("metric")},
            }
            print(converted_event)
            final_records.append(converted_event)

        import json

        with open(f"analysis/event_log.json", "w") as ef:
            json.dump(final_records, ef)

    # def event_monitoring(self, project=True):
    #     records = defaultdict(lambda: [])
    #
    #     for event in self._expr.find(
    #             filter={"type": DocType.Report.value},
    #             projection={field: 1 for field in self._throughput_projection_list}
    #             if project
    #             else None,
    #     ):
    #         records[event.get("id")].append(event)
    #
    #     import json
    #     for cid, events in records.items():
    #         emap = {}
    #         aggregated_events = []
    #         for event in events:
    #             eid = event.get("event")
    #             rec = emap.pop(eid, None)
    #             if rec is not None:
    #                 assert rec.get("status") == "start", "potential collision detected"
    #                 event_item = {}
    #                 event_item.update({
    #                     "eid": event.get("event"),
    #                     "start": rec.get("time"),
    #                     "end": event.get("time"),
    #                 })
    #                 aggregated_events.append(event_item)
    #             else:
    #                 emap.update({eid: event})
    #
    #         for k, v in emap.items():
    #             aggregated_events.append({
    #                 "eid": k,
    #                 "start": v.get("time"),
    #                 "end": float("inf"),
    #             })
    #         aggregated_events.sort(key=lambda x: x.get("start"))
    #         try:
    #             os.mkdir("analysis")
    #         except Exception as e:
    #             pass
    #         with open(f"analysis/{cid}_event_log.json", "w") as ef:
    #             json.dump(aggregated_events, ef)

    def drop_database(self):
        self._conn.drop_database(self._db)

    def drop_experiment(self):
        self._db.drop_collection(self._expr)


class MongoVisualizer:
    import pyecharts

    def __init__(
        self,
        db_host=None,
        db_port=None,
        db_name="expr",
        expr_name="",
        render_freq=10,
        port=8008,
    ):
        self._profiler = MongoProfiler(db_host, db_port, db_name, expr_name)
        self._render_freq = render_freq

        self._render_thread = None
        self._serving_port = port
        self._httpd = None
        self._http_thread = None

    def perf_visualize(self, graph, title, time, **kwargs):
        g = graph().add_xaxis(time)
        for dim_key, dim_data in kwargs.items():
            g = g.add_yaxis(
                dim_key,
                dim_data,
                label_opts=self.pyecharts.options.LabelOpts(is_show=False),
            )
        image = g.set_global_opts(
            xaxis_opts=self.pyecharts.options.AxisOpts(type_="value"),
            yaxis_opts=self.pyecharts.options.AxisOpts(type_="value"),
            title_opts=self.pyecharts.options.TitleOpts(title=title),
            datazoom_opts=self.pyecharts.options.DataZoomOpts(orient="horizontal"),
        )
        return image

    def metric_visualize(self, graph, title, data, align_with_time=True):
        g = graph()
        for plot_name, (times, steps, values) in data.items():
            if align_with_time:
                g.add_xaxis(times)
            else:
                g.add_xaxis(steps)
            g.add_yaxis(
                plot_name,
                values,
                label_opts=self.pyecharts.options.LabelOpts(is_show=False),
            )
        image = g.set_global_opts(
            xaxis_opts=self.pyecharts.options.AxisOpts(type_="value"),
            yaxis_opts=self.pyecharts.options.AxisOpts(type_="value"),
            title_opts=self.pyecharts.options.TitleOpts(title=title),
            datazoom_opts=self.pyecharts.options.DataZoomOpts(orient="horizontal"),
        )
        return image

    def render(self, metrics=True, resource=True, throughput=True, update_freq=None):
        update_freq = update_freq or self._render_freq
        print("Tracing meta-info")
        try:
            pprint.pprint(self._profiler.get_expr_meta_info())
        except Exception as e:
            print("Error detected in gathering meta information, exit")
            return

        while not getattr(threading.currentThread(), "exit_flag", False):
            if resource:
                page = self.pyecharts.charts.Page()
                resource_summaries = self._profiler.resource_monitoring()
                throughput_summaries = self._profiler.throughput_monitoring()
                for cid, res_summary in resource_summaries.items():
                    page = page.add(
                        self.perf_visualize(
                            self.pyecharts.charts.Line,
                            f"Resource\n{cid}",
                            **{
                                "time": (res_summary["time"]).tolist(),
                                "CPU(Percent)": res_summary["cpu"].tolist(),
                                "MEM(GB)": res_summary["mem"].tolist(),
                                **{
                                    f"GPU-{i}(GB)": res_summary["gpu"][:, i].tolist()
                                    for i in range(res_summary["gpu"].shape[1])
                                },
                            },
                        )
                    )

            if throughput:
                # page = self.pyecharts.charts.Page()
                throughput_summaries = self._profiler.throughput_monitoring()
                for cid, thp_summary in throughput_summaries.items():
                    page = page.add(
                        self.perf_visualize(
                            self.pyecharts.charts.Scatter,
                            f"throughput\n{cid}",
                            **{
                                "time": thp_summary["time"].tolist(),
                                "data entry(10^3)": thp_summary["data entry"].tolist(),
                                "exchange scale(MB)": thp_summary[
                                    "exchange scale"
                                ].tolist(),
                            },
                        )
                    )
                # page.render("throughput_profiler.html")

            if metrics:
                metric_summaries = self._profiler.metric_monitoring()
                for graph_name, graph_data in metric_summaries.items():
                    page = page.add(
                        self.metric_visualize(
                            self.pyecharts.charts.Line,
                            f"metric\n{graph_name}",
                            graph_data,
                        )
                    )

            page.render("profiler.html")
            time.sleep(update_freq)

    def start(self):
        self._render_thread = threading.Thread(
            target=self.render,
            args=(),
        )
        self._render_thread.daemon = True
        self._httpd = socketserver.TCPServer(
            ("", self._serving_port), ProfilingHtmlHandler
        )
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            args=(),
        )
        self._http_thread.daemon = True

        self._render_thread.start()
        print(f"Serving at http://localhost:{self._serving_port}")
        self._http_thread.start()

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._render_thread.exit_flag = True
        self._render_thread.join()
        self._http_thread.join()
        print("Rendering service has stopped")

    def __del__(self):
        try:
            self._render_thread.exit_flag = True

            self._httpd.shutdown()
            self._httpd.server_close()

            self._render_thread.join()
            self._http_thread.join()
            print("Rendering service has stopped")
        except Exception as e:
            pass

    def reset_all(self):
        self._profiler.drop_database()

    def reset(self):
        self._profiler.drop_experiment()
