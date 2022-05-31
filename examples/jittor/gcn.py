import jittor as jt
jt.flags.use_cuda = 1

from jittor import nn, Module, init
from jittor import optim
from jittor.contrib import slice_var_index

from tqdm import tqdm

from cogdl.layers.jittor import GCNLayer
from cogdl.datasets.planetoid_data import CoraDataset


def tensor2jit(x):
    return jt.array(x.cpu().numpy())

class GCN(Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.conv1 = GCNLayer(in_feats, hidden_size, dropout=dropout, activation="relu")
        self.conv2 = GCNLayer(hidden_size, out_feats)

    def execute(self, graph):
        graph.sym_norm()
        x = tensor2jit(graph.x)
        out = self.conv1(graph, x)
        out = self.conv2(graph, out)
        return out


def train(model, dataset):
    graph = dataset[0]

    optimizer = nn.AdamW(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    
    train_mask = tensor2jit(graph.train_mask)
    test_mask = tensor2jit(graph.test_mask)
    val_mask = tensor2jit(graph.val_mask)
    labels = tensor2jit(graph.y)
    
    for epoch in range(100):
        model.train()
        output = model(graph)
        loss = loss_function(output[train_mask], labels[train_mask])
        optimizer.step(loss)
        
        model.eval()
        with jt.no_grad():
            output = model(graph)
            pred = output.argmax(1)[0]
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        print(f"Epoch:{epoch}, loss:{loss:.3f}, val_acc:{val_acc:.3f}, test_acc:{test_acc:.3f}")

if __name__ == "__main__":
    dataset = CoraDataset()
    model = GCN(in_feats=dataset.num_features, hidden_size=64, out_feats=dataset.num_classes, dropout=0.5)

    train(model, dataset)
