from cogdl.utils import dropout_adj

import os.path as osp
import os

import argparse

import numpy as np

import torch

"""
The Following code is borrowed from SelfGNN
"""


class Augmentation:

    def __init__(self, p_f1=0.2, p_f2=0.1, p_e1=0.2, p_e2=0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"
    
    def _feature_masking(self, data, device):
        feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        x1, x2 = data.x.clone(), data.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, drop_rate=self.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, drop_rate=self.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.x, new_data2.x = x1, x2
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr , new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def __call__(self, data):
        
        return self._feature_masking(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data",
                        help="Path to data directory, where all the datasets will be placed. Default is 'data'")
    parser.add_argument("--name", "-n", type=str, default="WikiCS",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, WikiCS, and physics")
    parser.add_argument("--layers", "-l", nargs="+", default=[
                        512, 256], help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--pred_hid", '-ph', type=int,
                        default=512, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--init-parts", "-ip", type=int, default=1,
                        help="The number of initial partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--final-parts", "-fp", type=int, default=1,
                        help="The number of final partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--aug_params", "-p", nargs="+", default=[
                        0.1, 0.2, 0.4, 0.1], help="Hyperparameters for augmentation (p_f1, p_f2, p_e1, p_e2). Default is [0.2, 0.1, 0.2, 0.3]")
    parser.add_argument("--lr", '-lr', type=float, default=0.00001,
                        help="Learning rate. Default is 0.0001.")
    parser.add_argument("--warmup_epochs", '-we', type=int, default=1000,
                        help="Warmup epochs. Default is 1000.")
    parser.add_argument("--dropout", "-do", type=float,
                        default=0.0, help="Dropout rate. Default is 0.2")
    parser.add_argument("--cache-step", '-cs', type=int, default=10,
                        help="The step size to cache the model, that is, every cache_step the model is persisted. Default is 100.")
    parser.add_argument("--epochs", '-e', type=int,
                        default=20, help="The number of epochs")
    parser.add_argument("--device", '-d', type=int,
                        default=3, help="GPU to use")
    return parser.parse_args()


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None
        
        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]
            
            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)
            
            if hasattr(data, "train_mask") and data.train_mask is not None:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)
            else:
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
    
    else :
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
    
    return data