from typing import Dict

import numpy as np
import os
import torch
import torch.distributed as dist


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


def evaluation_comp(monitor, compare="<"):
    if "loss" in monitor or compare == "<":
        return np.inf, smaller_than
    else:
        return 0, bigger_than


def save_model(model, path, epoch):
    print(f"Saving {epoch}-th model to {path} ...")
    torch.save(model.state_dict(), path)
    model = model.model
    emb_path = os.path.dirname(path)
    if hasattr(model, "entity_embedding"):
        entity_embedding = model.entity_embedding.detach().numpy()
        print('Saving entity_embedding to ', path)
        np.save(os.path.join(emb_path, "entity_embedding"), entity_embedding) 

    if hasattr(model, "relation_embedding"):
        relation_embedding = model.relation_embedding.detach().numpy()
        print('Saving entity_embedding to ', path)
        np.save(os.path.join(emb_path, "relation_embedding"), relation_embedding)


def load_model(model, path):
    print(f"Loading model from {path} ...")
    model.load_state_dict(torch.load(path))
    return model


def ddp_after_epoch(*args):
    dist.barrier()


def ddp_end(*args):
    dist.barrier()
    dist.destroy_process_group()


class Printer(object):
    def __init__(self, print_fn, rank=0, world_size=1):
        self.printer = print_fn
        self.to_print = (world_size <= 1) or rank == 0 or rank == "cpu"

    def __call__(self, k_v: Dict):
        if self.to_print:
            assert "Epoch" in k_v
            out = f"Epoch: {k_v['Epoch']}"
            k_v.pop("Epoch")

            for k, v in k_v.items():
                if isinstance(v, float):
                    out += f", {k}: {v: .4f}"
                else:
                    out += f", {k}: {v}"
            self.printer(out)
