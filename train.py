import random
import numpy as np
import torch

from tqdm import tqdm
import torch.nn.functional as F

from cognitive_graph import options
from cognitive_graph.datasets import build_dataset
from cognitive_graph.models import build_model


def main(args):
    """Node classification task."""

    assert torch.cuda.is_available() and not args.cpu
    torch.cuda.set_device(args.device_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = build_dataset(args)
    data = dataset[0]
    data = data.cuda()
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    model = build_model(args)
    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    epoch_iter = tqdm(range(args.max_epoch))
    for epoch in epoch_iter:
        train_step(model, optimizer, data)
        epoch_iter.set_description(
            f"Epoch: {epoch:03d}, Train: {test_step(model, data, split='train'):.4f}, Val: {test_step(model, data, split='val'):.4f}"
        )
    test_acc = test_step(model, data, split="test")
    print(f"Test accuracy = {test_acc}")
    return test_acc


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(
        model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]
    ).backward()
    optimizer.step()


def test_step(model, data, split="val"):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    _, mask = list(data(f"{split}_mask"))[0]
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    main(args)
