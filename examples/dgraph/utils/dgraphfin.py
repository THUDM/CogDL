import torch
from cogdl.data import Graph
import numpy as np

def mask_change(id_mask, node_size):
    mask = torch.zeros(node_size).bool()
    for i in id_mask:
        mask[i] = True
    return mask


def Dgraph_Dataloader(datapath):
    # Load data
    print('read_dgraphfin')
    folder = datapath

    items = [np.load(folder)]

    # Create cogdl graph
    x = items[0]['x']
    y = items[0]['y'].reshape(-1, 1)

    edge_index = items[0]['edge_index']

    # set train/val/test mask in node_classification task
    train_id = items[0]['train_mask']
    valid_id = items[0]['valid_mask']
    test_id = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    # Feature normalization
    # x = (x - x.mean(0)) / x.std(0)

    y = torch.tensor(y, dtype=torch.int64)
    y = y.squeeze(1)

    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()

    # edge_type = torch.tensor(edge_type, dtype=torch.float)

    node_size = x.size()[0]

    train_m = torch.tensor(train_id, dtype=torch.int64)
    train_mask = mask_change(train_m, node_size)

    valid_m = torch.tensor(valid_id, dtype=torch.int64)
    valid_mask = mask_change(valid_m, node_size)

    test_m = torch.tensor(test_id, dtype=torch.int64)
    test_mask = mask_change(test_m, node_size)
        
    return x,edge_index,y,train_mask,valid_mask,test_mask
        

