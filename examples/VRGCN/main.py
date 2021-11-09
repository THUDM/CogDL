# ogb-arxiv accuracy: 72.3%

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloder import AdjSampler
from cogdl.datasets.ogb import OGBArxivDataset
from VRGCN import VRGCN

dataset = OGBArxivDataset(data_path='/home/tlhuang/dataset/cogdl')
data = dataset.data
data.add_remaining_self_loops()
data.set_symmetric()
# data.sym_norm()

evaluator = dataset.get_evaluator()
train_loader = AdjSampler(data, sizes=[2, 2], batch_size=2048, shuffle=True)
test_loader = AdjSampler(data, sizes=[-1], batch_size=4096, shuffle=False, training=False)

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
model = VRGCN(num_nodes=data.x.shape[0], in_channels=dataset.num_features,
              hidden_channels=256,
              out_channels=dataset.num_classes,
              num_layers=2, device=device).to(device)
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
    approx_acc = total_correct / torch.where(data['train_mask'])[0].size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    # out, _ = model.inference(x, data)
    out, _ = model.inference_batch(x, test_loader)
    
    y_true = y.cpu()
    y_pred = out.cpu()
    
    train_acc = evaluator(y_pred[data['train_mask']], y_true[data['train_mask']])
    val_acc = evaluator(y_pred[data['val_mask']], y_true[data['val_mask']])
    test_acc = evaluator(y_pred[data['test_mask']], y_true[data['test_mask']])

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1):
    model.reset_parameters()
    model.eval()
    model.initialize_history(x, test_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 100):
        loss, acc = train(epoch)
        if epoch % 1 == 0:
            train_acc, val_acc, test_acc = test()
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')