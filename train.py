import copy
import itertools
import random
from collections import defaultdict, namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from cognitive_graph import options
from cognitive_graph.tasks import build_task


def main(args):
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = build_task(args)
    return task.train()


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


if __name__ == "__main__":
    # Magic for making multiprocessing work for PyTorch
    mp.set_start_method("spawn")

    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_model_and_add_parameter(parser, args)
    print(args)
    variants = gen_variants(dataset=args.dataset, model=args.model, seed=args.seed)
    results_dict = defaultdict(list)

    device_ids = args.device_id
    if args.cpu:
        num_workers = len(args.dataset) * len(args.model * len(args.seed))
    else:
        num_workers = len(device_ids)
    print("num_workers", num_workers)

    with mp.Pool(processes=num_workers) as pool:
        for idx, variant in enumerate(variants):
            args.device_id = device_ids[idx % num_workers]
            args.dataset, args.model, args.seed = variant
            results_dict[variant[:-1]].append(
                pool.apply_async(func=main, args=(copy.copy(args),))
            )
        pool.close()
        pool.join()

    # Average for different seeds
    col_names = ["Variant"] + list(results_dict[variant[:-1]][-1].get().keys())

    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.get().values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))
