import sys
import os.path as osp

import torch
from cogdl.data import Data

def read_gatne_data(folder):
    train_data = {} 
    with open(osp.join(folder, '{}'.format('train.txt')), 'r') as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in train_data:
                train_data[items[0]] = []
            train_data[items[0]].append([int(items[1]), int(items[2])])

    valid_data = {} 
    with open(osp.join(folder, '{}'.format('valid.txt')), 'r') as f: 
        for line in f:
            items = line.strip().split()
            if items[0] not in valid_data:
                valid_data[items[0]] = [[], []]
            valid_data[items[0]][1-int(items[3])].append([int(items[1]), int(items[2])])
    
    test_data = {}
    with open(osp.join(folder, '{}'.format('test.txt')), 'r') as f: 
        for line in f:
            items = line.strip().split()
            if items[0] not in test_data:
                test_data[items[0]] = [[], []]
            test_data[items[0]][1-int(items[3])].append([int(items[1]), int(items[2])])

    data = Data()
    data.train_data = train_data
    data.valid_data = valid_data
    data.test_data = test_data
    return data
