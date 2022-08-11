from cogdl import experiment
from cogdl.datasets.kg_data import KnowledgeGraphDataset
import os.path as osp

# ./data/custom_dataset/raw need "train2id.txt", "valid2id.txt", "test2id.txt"
class Test_kgDatset(KnowledgeGraphDataset):
    def __init__(self, data_path="/home/cogdl/data"):
        dataset = "custom_dataset"
        path = osp.join(data_path, dataset)
        super((Test_kgDatset), self).__init__(path, dataset)

    def download(self):
        pass

if __name__ == "__main__":   
    dataset =Test_kgDatset()
    experiment(dataset=dataset, model="transe",do_valid=False,do_test=True,epochs=500,eval_step=501)
