import numpy as np

from cogdl import options
from cogdl.datasets import build_dataset_from_name


def download_datasets(args):
    if not isinstance(args.dataset, list):
        args.dataset = [args.dataset]

    for name in args.dataset:
        dataset = build_dataset_from_name(name)
        print(dataset[0])


if __name__ == "__main__":
    parser = options.get_download_data_parser()
    args = parser.parse_args()

    download_datasets(args)
