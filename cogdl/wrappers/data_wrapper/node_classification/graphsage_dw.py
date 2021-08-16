from .. import DataWrapper, register_data_wrapper
from cogdl.data.sampler import NeighborSampler


@register_data_wrapper("graphsage_dw")
class SAGEDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        # fmt: on

    def __init__(self, dataset, batch_size: int, sample_size: list):
        super(SAGEDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.x = self.dataset.data.x
        self.y = self.dataset.data.y
        self.batch_size = batch_size
        self.sample_size = sample_size

    def training_wrapper(self):
        self.dataset.data.train()
        return NeighborSampler(
            dataset=self.dataset,
            mask=self.dataset.data.train_mask,
            sizes=self.sample_size,
            num_workers=4,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def val_wrapper(self):
        self.dataset.data.eval()

        return NeighborSampler(
            dataset=self.dataset,
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
                dataset=self.dataset,
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
