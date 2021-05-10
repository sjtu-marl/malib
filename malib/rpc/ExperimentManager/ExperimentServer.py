import os
import time
import grpc
import multiprocessing
import traceback
import numpy as np
from signal import signal, SIGTERM
from concurrent import futures
from threading import Lock
from readerwriterlock import rwlock
from typing import Any, Dict

import tensorboardX

from malib.rpc.chunk import deserialize, recv_chunks, deserialize_image
from malib.rpc.proto import exprmanager_pb2, exprmanager_pb2_grpc
from malib.utils.convert import (
    utc_to_str,
    dump_dict,
    grpc_struct_to_dict,
    tensor_to_dict,
)


class _ConcurrentTable:
    def __init__(self):
        self.lock = rwlock.RWLockFair()
        # {name, idx}, {idx, writer}
        self.table = [{}, {}]

    def close(self):
        for idx, (lock, writer) in self.table[1].items():
            with lock:
                writer.close()

    def put(self, name):
        idx = -1
        with self.lock.gen_rlock():
            if name in self.table[0]:
                idx = self.table[0][name]

        if idx == -1:
            with self.lock.gen_wlock():
                idx = len(self.table[0])
                self.table[0][name] = idx
                writer = tensorboardX.SummaryWriter(name)
                self.table[1][idx] = (Lock(), writer)
        return idx

    def get(self, index):
        with self.lock.gen_rlock():
            wlock, writer = self.table[1][index]
        return wlock, writer


class ExperimentManagerRPCServicer(exprmanager_pb2_grpc.ExperimentManagerRPCServicer):
    def __init__(
        self,
        global_writer_table,
        logdir="./",
        flush_freq: int = -1,
        debug=False,
        verbose=False,
    ):
        super().__init__()

        self.root_dir = logdir
        self.table = global_writer_table

        self.debug = debug
        self.verbose = verbose

    def CreateTable(self, table_name, context):
        if self.debug:
            print(
                "Get CreateTable Request:\n", dump_dict(grpc_struct_to_dict(table_name))
            )

        rec_path = os.path.join(self.root_dir, table_name.primary, table_name.secondary)
        try:
            os.makedirs(rec_path)
        except Exception as e:
            if self.verbose:
                print("Error detected in making directory ", e)

        idx = -1
        try:
            idx = self.table.put(rec_path)
        except:
            traceback.print_exc()
        return exprmanager_pb2.TableKey(key=idx, time=time.time())

    def SendText(self, text, context):
        if self.debug:
            print("Get SendText Request:\n", text.text)
        try:
            lock, writer = self.table.get(text.key)
            with lock:
                writer.add_text(
                    text.tag, text.text, global_step=text.step, walltime=text.time
                )
            return exprmanager_pb2.SendReply(status=1, time=time.time())
        except Exception as e:
            if self.verbose:
                print("InternalError detected:", e)
            return exprmanager_pb2.SendReply(status=0, time=time.time())

    def SendScalar(self, scalar, context):
        if self.debug:
            print("Get SendScalar Request:\n", dump_dict(grpc_struct_to_dict(scalar)))

        try:
            lock, writer = self.table.get(scalar.key)
            scalar_type = scalar.WhichOneof("ScalarType")
            val = getattr(scalar, scalar_type)
            with lock:
                writer.add_scalar(
                    scalar.tag, val, global_step=scalar.step, walltime=scalar.time
                )
            return exprmanager_pb2.SendReply(status=1, time=time.time())
        except Exception as e:
            if self.verbose:
                print("InternalError detected:", e)
            return exprmanager_pb2.SendReply(status=0, time=time.time())

    def SendImage(self, binary_iterator, context):
        if self.debug:
            print("Get SendImage Request")

        try:
            serial_img, fields = recv_chunks(binary_iterator, "blocks")
            lock, writer = self.table.get(fields["key"])
            # print(fields["tensor"])
            if "tensor" in fields and not fields["tensor"]:
                img = deserialize_image(serial_img)
                img = np.array(img)
            else:
                img = deserialize(serial_img)
            if self.debug:
                print(img.shape)
            img = np.transpose(img[:, :, 0:3], [2, 0, 1])
            with lock:
                writer.add_image(
                    tag=fields["tag"],
                    img_tensor=img,
                    global_step=fields["step"],
                    walltime=fields["time"],
                )
            return exprmanager_pb2.SendReply(status=1, time=time.time())
        except Exception as e:
            if self.verbose:
                print(traceback.format_exc())
                print("InternalError detected:", e)
            return exprmanager_pb2.SendReply(status=0, time=time.time())

    def SendObj(self, binary_iterator, context):
        if self.debug:
            print("Get SendObj Request")
            print("Receiver currently act as method: send scalars")

        try:
            serial_obj, fields = recv_chunks(binary_iterator, "blocks")
            obj = deserialize(serial_obj)
            lock, writer = self.table.get(fields["key"])

            # Received internal info, currently only payoff
            if fields["tag"] == "__Payoff__":
                with lock:
                    writer.add_text(
                        "payoff_update_info",
                        "\n".join(
                            [
                                f"Update-SendTime-{utc_to_str(fields['time'])}:\n",
                                "* Population:\n",
                                "| AgentName |"
                                + "".join(
                                    [
                                        f" {agent} |"
                                        for agent, _ in obj["Population"].items()
                                    ]
                                ),
                                "| :-: |"
                                + "".join(
                                    [" :-: |" for _, _ in obj["Population"].items()]
                                ),
                                "| Policy |"
                                + "".join(
                                    [
                                        f" {pid} |"
                                        for _, pid in obj["Population"].items()
                                    ]
                                )
                                + "\n",
                                "* Reward:\n",
                                "| AgentName | "
                                + "".join(
                                    f" {agent} |" for (agent, _) in obj["Agents-Reward"]
                                ),
                                "| :-: |"
                                + "".join([" :-: |" for _, _ in obj["Agents-Reward"]]),
                                "| Reward |"
                                + "".join(
                                    f" {reward} |"
                                    for (_, reward) in obj["Agents-Reward"]
                                )
                                + "\n",
                            ]
                        ),
                        global_step=fields["step"],
                        walltime=fields["time"],
                    )
            else:

                def _flatten_obj(obj: Dict):
                    res = {}
                    for k, v in obj.items():
                        if isinstance(v, Dict):
                            temp_dict = _flatten_obj(v)
                            for tk, tv in temp_dict.items():
                                res[f"{k}/{tk}"] = tv
                        elif isinstance(v, float):
                            res[k] = v
                        else:
                            raise NotImplementedError
                    return res

                with lock:
                    # TODO(ming): flatten obj key
                    obj = _flatten_obj(obj)
                    for k, v in obj.items():
                        writer.add_scalar(
                            f"{fields['tag']}/{k}",
                            v,
                            global_step=fields["step"],
                            walltime=fields["time"],
                        )
            return exprmanager_pb2.SendReply(status=1, time=time.time())
        except Exception as e:
            if self.verbose:
                print("InternalError detected:", e)
            return exprmanager_pb2.SendReply(status=0, time=time.time())

    def SendBinaryTensor(self, binary, context):
        """
        Receive a tensor sent over rpc connection.
        In current implementation, the tensor is only printed in command shell
        since tensorboardX does not support adding tensors.
        Future: needed in sacred version experiment manager

        Parameters:import matplotlib.backends.backend_agg as plt_backend_agg
            binary: received binary rpc structs as predefined in exprmanager.proto
            context: rpc context
        """

        if self.debug:
            print("Get SendBinaryTensor Request")
        try:
            serial_tensor, key = recv_chunks(binary, "blocks")
            tensor = deserialize(serial_tensor)

            if self.debug:
                field_description = grpc_struct_to_dict(binary, skip_fields=["blocks"])
                print(field_description.update(tensor_to_dict(tensor)))

            return exprmanager_pb2.SendReply(status=1, time=time.time())
        except Exception as e:
            if self.verbose:
                print("InternalError detected:", e)
            return exprmanager_pb2.SendReply(status=0, time=time.time())


class ExprManagerServer:
    table = None

    def __init__(
        self, port, logdir="./", grace=5, max_workers=10, debug=False, verbose=False
    ):
        self.port = port
        self.grace = grace
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        self.table = _ConcurrentTable()
        exprmanager_pb2_grpc.add_ExperimentManagerRPCServicer_to_server(
            ExperimentManagerRPCServicer(
                global_writer_table=self.table,
                logdir=logdir,
                debug=debug,
                verbose=verbose,
            ),
            self.server,
        )
        self.server.add_insecure_port(self.port)

    def start(self):
        self.server.start()

    def wait(self):
        self.server.wait_for_termination()

    def stop(self):
        self.server.stop(grace=self.grace)
        self.table.close()


def _create_logging(**kwargs):
    s = ExprManagerServer(**kwargs)
    s.start()
    print("Logger server up!")

    def terminate_server(*_):
        s.stop()
        print("Logging server stop")

    signal(SIGTERM, terminate_server)
    s.wait()


def start_logging_server(**kwargs):
    process = multiprocessing.Process(target=_create_logging, kwargs=kwargs)
    return process
