from .. import DataWrapper
from cogdl.datasets.kg_data import BidirectionalOneShotIterator, TestDataset, TrainDataset
from torch.utils.data import DataLoader


class TripleDataWrapper(DataWrapper):
    @classmethod
    def add_args(self, parser):
        # fmt: off
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--test_batch_size", type=int, default=4)
        self.parser = parser
        return self.parser
        # fmt: on

    def __init__(self, dataset):
        super(TripleDataWrapper, self).__init__(dataset)
        self.args = self.parser.parse_args()
        self.dataset = dataset
        self.negative_sample_size = self.args.negative_sample_size
        self.batch_size = self.args.batch_size

    def train_wrapper(self):
        dataset = self.dataset

        train_iter = self.output_iter(dataset, self.negative_sample_size, self.batch_size)
        return train_iter

    def val_wrapper(self):
        dataset = self.dataset
        train_triples = dataset.triples[dataset.train_start_idx : dataset.valid_start_idx]

        valid_triples = dataset.triples[dataset.valid_start_idx : dataset.test_start_idx]
        test_triples = dataset.triples[dataset.test_start_idx :]
        all_true_triples = train_triples + valid_triples + test_triples
        test_dataloader_head = DataLoader(
            TestDataset(valid_triples, all_true_triples, dataset.num_entities, dataset.num_relations, "head-batch"),
            batch_size=self.args.test_batch_size,
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(valid_triples, all_true_triples, dataset.num_entities, dataset.num_relations, "tail-batch"),
            batch_size=self.args.test_batch_size,
            collate_fn=TestDataset.collate_fn,
        )

        return (test_dataloader_head, test_dataloader_tail)

    def test_wrapper(self):
        dataset = self.dataset
        train_triples = dataset.triples[dataset.train_start_idx : dataset.valid_start_idx]

        valid_triples = dataset.triples[dataset.valid_start_idx : dataset.test_start_idx]
        test_triples = dataset.triples[dataset.test_start_idx :]
        all_true_triples = train_triples + valid_triples + test_triples
        test_dataloader_head = DataLoader(
            TestDataset(test_triples, all_true_triples, dataset.num_entities, dataset.num_relations, "head-batch"),
            batch_size=self.args.test_batch_size,
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(test_triples, all_true_triples, dataset.num_entities, dataset.num_relations, "tail-batch"),
            batch_size=self.args.test_batch_size,
            collate_fn=TestDataset.collate_fn,
        )
        return (test_dataloader_head, test_dataloader_tail)

    @staticmethod
    def output_iter(dataset, negative_sample_size, batch_size):
        train_triples = dataset.triples[dataset.train_start_idx : dataset.valid_start_idx]
        nentity, nrelation = dataset._num_entities, dataset._num_relations

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, "head-batch"),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, "tail-batch"),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn,
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        return train_iterator
