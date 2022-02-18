import time
import grpc
import logging
import builtins
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from malib.rpc.chunk import serialize, binary_chunks
from malib.rpc.proto import exprmanager_pb2, exprmanager_pb2_grpc
from malib.rpc.chunk import serialize_pyplot
from malib.utils.configs.config import EXPERIMENT_MANAGER_CONFIG as CONFIG


class ExprManagerClient:
    def __init__(self, server_addr: str, nid: str, log_level: int, *args, **kwargs):
        """Initialize an independent experiment logging client.

        :param str server_addr: The experiment server address
        :param str nid: Client's human readable name
        :param int log_level: Logging level
        :param tuple args: A tuple of args
        :param dict kwargs: A dict of args
        """

        self._server_addr = server_addr
        self._executor = ThreadPoolExecutor(max_workers=10)

        # a copy of default experiment manager config
        self._config = CONFIG.copy()
        # update name and logging level
        self._config["nid"] = nid or ""
        self._config["log_level"] = log_level
        self.is_remote = True

    @staticmethod
    def _chunk_generator_wrapper(
        key, tag, src, tensor, step, time, internal_generator, wrapper_func, encode=True
    ):
        for item in internal_generator:
            yield wrapper_func(
                key=key,
                tag=tag,
                src=str(src),
                tensor=tensor,
                blocks=item,
                step=int(step),
                time=float(time),
            )

    @staticmethod
    def _create_table(server_addr, primary, secondary, nid, **kwargs):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)
            table_name = exprmanager_pb2.TableName(
                primary=primary, secondary=secondary, src=str(nid), time=time.time()
            )
            table_key = stub.CreateTable(table_name, **kwargs)
            return table_key.key, table_key.time

    def _create_table_callback(self, key):
        self._config["key"] = key

    def create_table(
        self, primary=None, secondary=None, nid=None, blocking=False, *args, **kwargs
    ):
        """Create a logging table.

        :param str primary: None,
        :param str secondary: None,
        :param str nid: None
        :param bool blocking: None
        """

        _call_params = {
            "server_addr": self._server_addr,
            "primary": self._config["primary"] if primary is None else primary,
            "secondary": self._config["secondary"] if secondary is None else secondary,
            "nid": self._config["nid"] if nid is None else nid,
        }

        if blocking:
            table_key, res_time = self._create_table(**_call_params, **kwargs)
            self._create_table_callback(table_key)
            return table_key, res_time
        else:
            future = self._executor.submit(self._create_table, **_call_params, **kwargs)

            # for the safe of the data, create_table is now blocking
            while not future.done():
                continue
            self._create_table_callback(future.result()[0])
            return future

    @staticmethod
    def _send_text(server_addr, key, tag, nid, content, step, time, **kwargs):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)
            text = exprmanager_pb2.Text(
                key=key,
                tag=tag,
                src=str(nid),
                text=str(content),
                step=int(step),
                time=float(time),
            )
            response = stub.SendText(text, **kwargs)
            return response.status, response.time

    def send_text(
        self,
        key=None,
        tag="",
        nid=None,
        content="",
        global_step=None,
        walltime=None,
        blocking=False,
        *args,
        **kwargs,
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "content": content,
            "step": 0 if global_step is None else global_step,
            "time": time.time() if walltime is None else walltime,
        }

        if blocking:
            return self._send_text(**_call_params, **kwargs)
        else:
            future = self._executor.submit(self._send_text, **_call_params, **kwargs)
            return future

    def info(self, content="", blocking=False, **kwargs):
        if self._config["log_level"] > logging.INFO:
            return None

        return self.send_text(
            key=self._config["key"],
            tag="info",
            nid=self._config["nid"],
            content=content,
            blocking=blocking,
            **kwargs,
        )

    def warning(self, content="", blocking=False, **kwargs):
        if self._config["log_level"] > logging.WARNING:
            return None

        return self.send_text(
            key=self._config["key"],
            tag="warning",
            nid=self._config["nid"],
            content=content,
            blocking=blocking,
            **kwargs,
        )

    def debug(self, content="", blocking=False, **kwargs):
        if self._config["log_level"] > logging.DEBUG:
            return None

        return self.send_text(
            key=self._config["key"],
            tag="debug",
            nid=self._config["nid"],
            content=content,
            blocking=blocking,
            **kwargs,
        )

    def error(self, content="", **kwargs):
        if self._config["log_level"] > logging.ERROR:
            return None

        return self._send_text(
            key=self._config["key"],
            tag="error",
            nid=self._config["nid"],
            content=content,
            **kwargs,
        )

    def critical(self, content="", **kwargs):
        if self._config["log_level"] > logging.CRITICAL:
            return None

        return self._send_text(
            key=self._config["key"],
            tag="critical",
            nid=self._config["nid"],
            content=content,
            **kwargs,
        )

    def exception(self, content="", **kwargs):
        if self._config["log_level"] > logging.ERROR:
            return None

        return self._send_text(
            key=self._config["key"],
            tag="exception",
            nid=self._config["nid"],
            content=content,
            **kwargs,
        )

    @staticmethod
    def _send_scalar(server_addr: str, key: str, tag, nid, content, step, time):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)
            # FIXME(ming): blocked here
            scalar = exprmanager_pb2.Scalar(
                key=key, tag=tag, src=str(nid), step=int(step), time=float(time)
            )
            if isinstance(content, np.integer) or isinstance(content, np.floating):
                content = content.item()
            content_type_name = content.__class__.__name__
            op_name = "op_" + content_type_name
            if hasattr(scalar, op_name):
                setattr(scalar, op_name, content)
            else:
                print(
                    f"Unknown type: {content_type_name} can not be converted to scalar."
                )
            response = stub.SendScalar(scalar)
            return response.status, response.time

    def send_scalar(
        self,
        key=None,
        tag="",
        nid=None,
        content=None,
        global_step=None,
        walltime=None,
        blocking=False,
        *args,
        **kwargs,
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "content": content,
            "step": 0 if global_step is None else global_step,
            "time": time.time() if walltime is None else walltime,
        }

        if blocking:
            status = self._send_scalar(**_call_params, **kwargs)
            return status
        else:
            future = self._executor.submit(self._send_scalar, **_call_params, **kwargs)
            return future

    def send_scalars(self, key=None, tag="", nid=None, content=None, *args, **kwargs):
        """Send multiple scalars to log server

        Parameters
        ----------
        key
            str, the logging table name
        tag
            str, tag
        nid
            str, ???
        content
            Any, logging data, or must it be a dict ?
        args
        kwargs

        Returns
        -------

        """
        return self.send_obj(key=key, tag=tag, nid=nid, obj=content, *args, **kwargs)

    @staticmethod
    def _send_image(server_addr, key, tag, nid, image, step, time, serial=True):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)

            if serial:
                image = serialize(image)

            image_chunk_generator = binary_chunks(serial_obj=image)
            wrapper_generator = ExprManagerClient._chunk_generator_wrapper(
                key=key,
                tag=tag,
                src=nid,
                tensor=serial,
                step=step,
                time=time,
                internal_generator=image_chunk_generator,
                wrapper_func=exprmanager_pb2.Binary,
            )
            response = stub.SendImage(wrapper_generator)
            return response.status, response.time

    def send_image(
        self,
        key=None,
        tag="",
        nid=None,
        image=None,
        global_step=None,
        walltime=None,
        serial=True,
        blocking=False,
        *args,
        **kwargs,
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "image": image,
            "serial": serial,
            "step": 0 if global_step is None else global_step,
            "time": time.time() if walltime is None else walltime,
        }

        if blocking:
            return self._send_image(**_call_params)
        else:
            return self._executor.submit(self._send_image, **_call_params)

    def send_figure(
        self,
        key=None,
        tag="",
        nid=None,
        figure=None,
        global_step=None,
        walltime=None,
        blocking=False,
        *args,
        **kwargs,
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "image": serialize_pyplot(figure=figure),
            "serial": False,
            "step": 0 if global_step is None else global_step,
            "time": time.time() if walltime is None else walltime,
        }

        if blocking:
            return self._send_image(**_call_params)
        else:
            return self._executor.submit(self._send_image, **_call_params)

    @staticmethod
    def _send_obj(server_addr, key, tag, nid, obj, step, time, serial=True):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)
            if serial:
                serial_obj = serialize(obj)
            else:
                serial_obj = obj
            chunk_generator = binary_chunks(serial_obj)
            wrapper_generator = ExprManagerClient._chunk_generator_wrapper(
                key=key,
                tag=tag,
                src=nid,
                tensor=serial,
                step=step,
                time=time,
                internal_generator=chunk_generator,
                wrapper_func=exprmanager_pb2.Binary,
            )
            response = stub.SendObj(wrapper_generator)
            return response.status, response.time

    def send_obj(
        self,
        key=None,
        tag="",
        nid=None,
        obj=None,
        global_step=None,
        walltime=None,
        serial=True,
        blocking=False,
        *args,
        **kwargs,
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "obj": obj,
            "serial": serial,
            "step": 0 if global_step is None else global_step,
            "time": time.time() if walltime is None else walltime,
        }

        if blocking:
            return self._send_obj(**_call_params)
        else:
            return self._executor.submit(self._send_obj, **_call_params)

    @staticmethod
    def _send_binary_tensor(server_addr, key, tag, nid, tensor, global_step, walltime):
        with grpc.insecure_channel(server_addr) as channel:
            stub = exprmanager_pb2_grpc.ExperimentManagerRPCStub(channel)
            chunk_generator = binary_chunks(serialize(tensor))
            wrapper_generator = ExprManagerClient._chunk_generator_wrapper(
                key=key,
                tag=tag,
                src=nid,
                step=0,
                time=time.time(),
                internal_generator=chunk_generator,
                wrapper_func=exprmanager_pb2.Binary,
            )
            response = stub.SendBinaryTensor(wrapper_generator)
            return response.status, response.time

    def send_binary_tensor(
        self, key=None, tag="", nid=None, tensor=None, blocking=False, *args, **kwargs
    ):
        _call_params = {
            "server_addr": self._server_addr,
            "key": self._config["key"] if key is None else key,
            "tag": tag,
            "nid": self._config["nid"] if nid is None else nid,
            "tensor": tensor,
        }

        if blocking:
            return self._send_binary_tensor(**_call_params)
        else:
            return self._executor.submit(self._send_binary_tensor, **_call_params)

    def report(self, **kwargs):
        return None
