from .. import DataWrapper
from cogdl.data.sampler import ClusteredLoader, ClusteredDataset


class ClusterWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--n-cluster", type=int, default=100)
        parser.add_argument("--method", type=str, default="metis")
        # fmt: on

    def __init__(self, dataset, method="metis", batch_size=20, n_cluster=100):
        super(ClusterWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.cluster_dataset = ClusteredDataset(dataset, n_cluster=n_cluster, batch_size=batch_size)
        self.batch_size = batch_size
        self.n_cluster = n_cluster
        self.method = method

    def train_wrapper(self):
        self.dataset.data.train()
        return ClusteredLoader(
            self.cluster_dataset,
            method=self.method,
            batch_size=self.batch_size,
            shuffle=True,
            n_cluster=self.n_cluster,
            # persistent_workers=True,
            num_workers=0,
        )

    def get_train_dataset(self):
        return self.cluster_dataset

    def val_wrapper(self):
        self.dataset.data.eval()
        return self.dataset.data

    def test_wrapper(self):
        self.dataset.data.eval()
        return self.dataset.data
