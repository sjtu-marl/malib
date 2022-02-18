# import pytest
# import numpy as np

# from malib.rollout import rollout_func


# def test_eval_info_parser():
#     ph = []
#     holder = {}
#     for history, ds, k, vs in rollout_func.iter_many_dicts_recursively(*ph, history=[]):
#         arr = [np.sum(_vs) for _vs in vs]
#         prefix = "/".join(history)
#         holder[prefix] = arr
