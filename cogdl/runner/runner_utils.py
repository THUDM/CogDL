import numpy as np

import torch


def merge_batch_indexes(outputs: list):
    assert len(outputs) > 0
    keys = list(outputs[0].keys())

    results = dict()
    for key in keys:
        values = [x[key] for x in outputs]
        if key.endswith("loss"):
            results[key] = sum(values).item() / len(values)
        elif key.endswith("eval_index"):
            if len(values) > 1:
                val = torch.cat(values, dim=0)
                val = val.sum(0)
            else:
                val = values[0]
            fp = val[0]
            all_ = val.sum()

            prefix = key[: key.find("eval_index")]
            if val.shape[0] == 2:
                _key = prefix + "acc"
            else:
                _key = prefix + "f1"
            results[_key] = (fp / all_).item()
        else:
            results[key] = sum(values)
    return results


def bigger_than(x, y):
    return x >= y


def smaller_than(x, y):
    return x <= y


def evaluation_comp(monitor):
    if "loss" in monitor:
        return np.inf, smaller_than
    else:
        return 0, bigger_than
