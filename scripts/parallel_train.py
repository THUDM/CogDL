import copy
import time
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.experiments import gen_variants
from cogdl.tasks import build_task
from cogdl.utils import set_random_seed, tabulate_results
from tabulate import tabulate


def main(args):
    if torch.cuda.is_available() and not args.cpu:
        pid = mp.current_process().pid
        torch.cuda.set_device(args.pid_to_cuda[pid])

    set_random_seed(args.seed)

    task = build_task(args)
    result = task.train()
    return result


def getpid(_):
    # HACK to get different pids
    time.sleep(1)
    return mp.current_process().pid


if __name__ == "__main__":
    # Magic for making multiprocessing work for PyTorch
    mp.set_start_method("spawn", force=True)

    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)

    # Make sure datasets are downloaded first
    datasets = args.dataset
    for dataset in datasets:
        args.dataset = dataset
        _ = build_dataset(args)
    args.dataset = datasets

    print(args)
    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed))

    device_ids = args.device_id
    if args.cpu:
        num_workers = 1
    else:
        num_workers = len(device_ids)
    print("num_workers", num_workers)

    results_dict = defaultdict(list)
    with mp.Pool(processes=num_workers) as pool:
        # Map process to cuda device
        pids = pool.map(getpid, range(num_workers))
        pid_to_cuda = dict(zip(pids, device_ids))

        # yield all variants
        def variant_args_generator():
            """Form variants as group with size of num_workers"""
            for variant in variants:
                args.pid_to_cuda = pid_to_cuda
                args.dataset, args.model, args.seed = variant
                yield copy.deepcopy(args)

        # Collect results
        results = pool.map(main, variant_args_generator())
        for variant, result in zip(variants, results):
            results_dict[variant[:-1]].append(result)

    # Average for different seeds
    col_names = ["Variant"] + list(results_dict[variants[0][:-1]][-1].keys())
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))
