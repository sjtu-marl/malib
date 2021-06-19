import time


def utc_to_str(utc_time) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(utc_time))


def dump_dict(d, indent=4):
    ss = "{\n"
    for k, v in d.items():
        ss += " " * indent + str(k) + ": "
        if isinstance(v, dict):
            ss += dump_dict(v, indent=indent + 4)
        else:
            ss += str(v)
        ss += "\n"
    ss += "}\n"
    return ss


def grpc_struct_to_dict(any_struct, skip_fields=[]):
    res = {}
    for f in any_struct.DESCRIPTOR.fields:
        if f not in skip_fields:
            res[f.name] = getattr(any_struct, f.name)
    return res


def tensor_to_dict(input_tensor):
    import numpy as np

    res = {}
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError("numpy.ndarray objects expected")
    res["Shape"] = input_tensor.shape
    res["Values"] = input_tensor
    return res


def anyof(dict_object: dict):
    return next(iter(dict_object.values()))
