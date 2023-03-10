from cogdl.data import Dataset

from cogdl.wrappers.data_wrapper.base_data_wrapper import OnLoadingWrapper


class DataController(object):
    def __init__(self, distributed: bool = False):
        self.distributed = distributed

    def prepare_data_wrapper(self, dataset_w):
        
        dataset_w.pre_transform()
        dataset_w.prepare_training_data()
        dataset_w.prepare_val_data()
        dataset_w.prepare_test_data()
        return dataset_w

    def training_proc_per_stage(self, dataset_w):
        if dataset_w.__refresh_per_epoch__(): 
            dataset_w.prepare_training_data()
        return dataset_w
