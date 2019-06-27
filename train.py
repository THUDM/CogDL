import copy
import itertools
import random
from collections import defaultdict, namedtuple

import numpy as np
import torch
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
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()

    variants = gen_variants(dataset=args.dataset, model=args.model, seed=args.seed)
    results_dict = defaultdict(list)
    for variant in variants:
        args.dataset, args.model, args.seed = variant
        # Parse *-specific arguments. *: model, task
        args = options.parse_args_and_arch(parser, args)
        # Reset arguments to variant
        args.dataset, args.model, args.seed = variant
        print(variant)
        result = main(args)
        results_dict[variant[:-1]].append(np.array(list(result.values())))

    col_names = ["Variant"] + list(result.keys())
    # Average for different seeds
    tab_data = []
    for variant in results_dict:
        results = results_dict[variant]
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
