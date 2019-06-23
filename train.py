import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cognitive_graph import options
from cognitive_graph.tasks import build_task


def main(args):
    assert torch.cuda.is_available() and not args.cpu
    torch.cuda.set_device(args.device_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = build_task(args)
    task.train(num_epoch=args.max_epoch)


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    main(args)
