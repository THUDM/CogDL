from .. import DataWrapper
from cogdl.data.sampler import NeighborSampler, NeighborSamplerDataset


class GraphSAGEDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        # fmt: on

    def __init__(self, dataset, batch_size: int, sample_size: list):
        super(GraphSAGEDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.train_dataset = NeighborSamplerDataset(
            dataset, sizes=sample_size, batch_size=batch_size, mask=dataset.data.train_mask
        )
        self.val_dataset = NeighborSamplerDataset(
            dataset, sizes=sample_size, batch_size=batch_size * 2, mask=dataset.data.val_mask
        )
        self.test_dataset = NeighborSamplerDataset(
            dataset=self.dataset, mask=None, sizes=[-1], batch_size=batch_size * 2,
        )
        self.x = self.dataset.data.x
        self.y = self.dataset.data.y
        self.batch_size = batch_size
        self.sample_size = sample_size

    def train_wrapper(self):
        self.dataset.data.train()
        return NeighborSampler(
            dataset=self.train_dataset,
            mask=self.dataset.data.train_mask,
            sizes=self.sample_size,
            num_workers=4,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def val_wrapper(self):
        self.dataset.data.eval()

        return NeighborSampler(
            dataset=self.val_dataset,
            mask=self.dataset.data.val_mask,
            sizes=self.sample_size,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=4,
        )

    def test_wrapper(self):
        return (
            self.dataset,
            NeighborSampler(
                dataset=self.test_dataset,
                mask=None,
                sizes=[-1],
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=4,
            ),
        )

    def train_transform(self, batch):
        target_id, n_id, adjs = batch
        x_src = self.x[n_id]
        y = self.y[target_id]
        return x_src, y, adjs

    def val_transform(self, batch):
        return self.train_transform(batch)

    def get_train_dataset(self):
        return self.train_dataset
