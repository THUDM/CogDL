import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataloder import AdjSampler
from tqdm import tqdm
from cogdl.datasets.ogb import OGBArxivDataset
from VRGCN import VRGCN


def get_parser():
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (CogDL GNNs)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-neighbors", type=list, default=[2, 2])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    return args


args = get_parser()

dataset = OGBArxivDataset(data_path="data/")
data = dataset.data
data.add_remaining_self_loops()
data.set_symmetric()
# data.sym_norm()

evaluator = dataset.get_evaluator()
train_loader = AdjSampler(data, sizes=args.num_neighbors, batch_size=args.batch_size, shuffle=True)
test_loader = AdjSampler(data, sizes=[-1], batch_size=args.batch_size, shuffle=False, training=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VRGCN(
    num_nodes=data.x.shape[0],
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_size,
    out_channels=dataset.num_classes,
    dropout=args.dropout,
    num_layers=args.num_layers,
    device=device,
).to(device)
model.reset_parameters()

x = data.x
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()
    total_loss = total_correct = 0
    for batch, sample_ids_adjs, full_ids_adjs in train_loader:
        optimizer.zero_grad()
        out = model(x, sample_ids_adjs, full_ids_adjs)
        loss = F.nll_loss(out, y[batch])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[batch]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / torch.where(data["train_mask"])[0].size(0)
    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out, _ = model.inference_batch(x, test_loader)

    y_true = y.cpu()
    y_pred = out.cpu()

    train_acc = evaluator(y_pred[data["train_mask"]], y_true[data["train_mask"]])
    val_acc = evaluator(y_pred[data["val_mask"]], y_true[data["val_mask"]])
    test_acc = evaluator(y_pred[data["test_mask"]], y_true[data["test_mask"]])
    return train_acc, val_acc, test_acc


test_accs = []
for run in range(args.runs):
    model.reset_parameters()
    model.eval()
    model.initialize_history(x, test_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = final_test_acc = 0
    epoch_iter = tqdm(range(1, args.epochs))
    for epoch in epoch_iter:
        loss, acc = train(epoch)
        if epoch % 1 == 0:
            train_acc, val_acc, test_acc = test()
            desc_dict = {
                "Run": run + 1,
                "Epoch": epoch,
                "Loss": f"{loss:.4f}",
                "Train": f"{100 * train_acc:.2f}%",
                "Valid": f"{100 * val_acc:.2f}%",
                "Test": f"{100 * test_acc:.2f}%",
            }
            epoch_iter.set_description(", ".join([key + ": " + str(value) for key, value in desc_dict.items()]))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print("============================")
print(f"Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}")
