import numpy as np

from typing import Dict


def compare(source, dest):
    for k, v in source.items():
        print(f"------------------check: {k}")
        v = v.data.numpy()
        d = dest[k].data.numpy()
        diff = np.mean(v - d)
        if np.isnan(diff):
            print(f"********** got non on: {k}:\n" f"max: {v.max()} {d.max()}")
            exit(1)
        else:
            print(f"----------- check diff on {k}: {np.mean(v - d)}")


def check_nan(_id, state_dict):
    if isinstance(state_dict, Dict):
        for k, v in state_dict.items():
            # print(f"----- check for {_id}/{k}:")
            check_nan(f"{_id}/{k}", v)
    else:
        v = state_dict.detach().numpy()
        # print(f"max: {v.max()} min: {v.min()}")
