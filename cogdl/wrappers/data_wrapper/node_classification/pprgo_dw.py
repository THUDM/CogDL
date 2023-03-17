import os
import scipy.sparse as sp
from cogdl import function as BF
from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from jittor.dataset import Dataset
    from jittor.dataset import Dataset as DataLoader
    from jittor.dataset import BatchSampler
    from jittor.dataset import SequentialSampler
elif BACKEND == "torch":
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from torch.utils.data import BatchSampler
    from torch.utils.data import SequentialSampler

from .. import DataWrapper
from cogdl.utils.ppr_utils import build_topk_ppr_matrix_from_data


class PPRGoDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--topk", type=int, default=32)
        parser.add_argument("--norm", type=str, default="sym")
        parser.add_argument("--eps", type=float, default=1e-4)

        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--test-batch-size", type=int, default=-1)
        # fmt: on

    def __init__(self, dataset, topk, alpha=0.2, norm="sym", batch_size=512, eps=1e-4, test_batch_size=-1):
        super(PPRGoDataWrapper, self).__init__(dataset)
        self.batch_size, self.test_batch_size = batch_size, test_batch_size
        self.topk, self.alpha, self.norm, self.eps = topk, alpha, norm, eps
        self.dataset = dataset

    def train_wrapper(self):
        """
        batch: tuple(x, targets, ppr_scores, y)
            x: shape=(b, num_features)
            targets: shape=(num_edges_of_batch,)
             ppr_scores: shape=(num_edges_of_batch,)
             y: shape=(b, num_classes)
        """
        self.dataset.data.train()
        ppr_dataset_train = pre_transform(self.dataset, self.topk, self.alpha, self.eps, self.norm, mode="train")
        train_loader = setup_dataloader(ppr_dataset_train, self.batch_size)
        return train_loader

    def val_wrapper(self):
        self.dataset.data.eval()
        if self.test_batch_size > 0:
            ppr_dataset_val = pre_transform(self.dataset, self.topk, self.alpha, self.eps, self.norm, mode="val")
            val_loader = setup_dataloader(ppr_dataset_val, self.test_batch_size)
            return val_loader
        else:
            return self.dataset.data

    def test_wrapper(self):
        self.dataset.data.eval()
        if self.test_batch_size > 0:
            ppr_dataset_test = pre_transform(self.dataset, self.topk, self.alpha, self.eps, self.norm, mode="test")
            test_loader = setup_dataloader(ppr_dataset_test, self.test_batch_size)
            return test_loader
        else:
            return self.dataset.data


def setup_dataloader(ppr_dataset, batch_size):

    data_loader = DataLoader(
        dataset=ppr_dataset,
        sampler=BatchSampler(
            SequentialSampler(ppr_dataset),
            batch_size=batch_size,
            drop_last=False,
        ),
        batch_size=None,
    )
    return data_loader


def pre_transform(dataset, topk, alpha, epsilon, normalization, mode="train"):
    dataset_name = dataset.__class__.__name__
    data = dataset[0]
    num_nodes = data.x.shape[0]
    nodes = BF.arange(num_nodes)

    mask = getattr(data, f"{mode}_mask")
    index = nodes[mask].numpy()
    if mode == "train":
        data.train()
    else:
        data.eval()
    edge_index = data.edge_index

    if not os.path.exists("./pprgo_saved"):
        os.mkdir("pprgo_saved")
    path = f"./pprgo_saved/{dataset_name}_{topk}_{alpha}_{normalization}.{mode}.npz"

    if os.path.exists(path):
        print(f"Load {mode} from cached")
        topk_matrix = sp.load_npz(path)
    else:
        print(f"Fail to load {mode}, generating...")
        topk_matrix = build_topk_ppr_matrix_from_data(edge_index, alpha, epsilon, index, topk, normalization)
        sp.save_npz(path, topk_matrix)
    result = PPRGoDataset(data.x, topk_matrix, index, data.y)
    return result


class PPRGoDataset(Dataset):
    def __init__(
        self,
        features: BF.dtype_dict("tensor"),  # noqa
        ppr_matrix: sp.csr_matrix,
        node_indices: BF.dtype_dict("tensor"),  # noqa
        labels_all: BF.dtype_dict("tensor") = None,  # noqa
    ):
        self.features = features
        self.matrix = ppr_matrix
        self.node_indices = node_indices
        self.labels_all = labels_all
        self.cache = dict()

    def __len__(self):
        return self.node_indices.shape[0]

    def __getitem__(self, items):
        key = str(items)
        if key not in self.cache:
            sample_matrix = self.matrix[items]
            source, neighbor = sample_matrix.nonzero()
            ppr_scores = BF.from_numpy(sample_matrix.data).float()

            features = self.features[neighbor].float()
            targets = BF.from_numpy(source).long()
            labels = self.labels_all[self.node_indices[items]]
            self.cache[key] = (features, targets, ppr_scores, labels)
        return self.cache[key]
