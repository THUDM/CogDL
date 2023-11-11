from cogdl.wrappers.data_wrapper import DataWrapper
from cogdl.wrappers.tools.wrapper_utils import node_degree_as_feature, split_dataset
from cogdl.data import DataLoader


class GraphClassificationDataWrapper(DataWrapper):
    def __init__(self, dataset, batch_size, num_workers, collate_fn=None):
        super(GraphClassificationDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def train_wrapper(self):
        return DataLoader(self.dataset.get_subset(self.dataset.split_index["train"]), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=self.collate_fn)
    
    def val_wrapper(self):
        return DataLoader(self.dataset.get_subset(self.dataset.split_index["valid"]), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fn)
    
    def test_wrapper(self):
        return DataLoader(self.dataset.get_subset(self.dataset.split_index["test"]),  batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fn)
    
    def num_iterations(self):
        return len(self.train_wrapper())
