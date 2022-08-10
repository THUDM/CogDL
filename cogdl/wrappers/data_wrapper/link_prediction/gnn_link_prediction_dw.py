import numpy as np
import torch
from .. import DataWrapper


class GNNLinkPredictionDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(GNNLinkPredictionDataWrapper, self).__init__(dataset)
        self.dataset = dataset

    def train_wrapper(self):
        return self.dataset.data

    def val_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data

    def pre_transform(self):
        data = self.dataset.data
        num_nodes = data.x.shape[0]
        ((train_edges, val_edges, test_edges), (val_false_edges, test_false_edges),) = self.train_test_edge_split(
            data.edge_index, num_nodes
        )
        data.train_edges = train_edges
        data.val_edges = val_edges
        data.test_edges = test_edges
        data.val_neg_edges = val_false_edges
        data.test_neg_edges = test_false_edges
        self.dataset.data = data

    @staticmethod
    def train_test_edge_split(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.2):
        row, col = edge_index
        mask = row > col
        row, col = row[mask], col[mask]
        num_edges = row.size(0)

        perm = torch.randperm(num_edges)
        row, col = row[perm], col[perm]

        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)

        index = [[0, num_val], [num_val, num_val + num_test], [num_val + num_test, -1]]
        sampled_rows = [row[l:r] for l, r in index]  # noqa E741
        sampled_cols = [col[l:r] for l, r in index]  # noqa E741

        # sample false edges
        num_false = num_val + num_test
        row_false = np.random.randint(0, num_nodes, num_edges * 5)
        col_false = np.random.randint(0, num_nodes, num_edges * 5)

        indices_false = row_false * num_nodes + col_false
        indices_true = row.cpu().numpy() * num_nodes + col.cpu().numpy()
        indices_false = list(set(indices_false).difference(indices_true))
        indices_false = np.array(indices_false)
        row_false = indices_false // num_nodes
        col_false = indices_false % num_nodes

        mask = row_false > col_false
        row_false = row_false[mask]
        col_false = col_false[mask]

        edge_index_false = np.stack([row_false, col_false])
        if edge_index[0].shape[0] < num_false:
            ratio = edge_index_false.shape[1] / num_false
            num_val = int(ratio * num_val)
            num_test = int(ratio * num_test)
        val_false_edges = torch.from_numpy(edge_index_false[:, 0:num_val])
        test_fal_edges = torch.from_numpy(edge_index_false[:, num_val : num_test + num_val])

        def to_undirected(_row, _col):
            _edge_index = torch.stack([_row, _col], dim=0)
            _r_edge_index = torch.stack([_col, _row], dim=0)
            return torch.cat([_edge_index, _r_edge_index], dim=1)

        train_edges = to_undirected(sampled_rows[2], sampled_cols[2])
        val_edges = torch.stack([sampled_rows[0], sampled_cols[0]])
        test_edges = torch.stack([sampled_rows[1], sampled_cols[1]])
        return (train_edges, val_edges, test_edges), (val_false_edges, test_fal_edges)
