import os.path as osp
import os
import sys
import json
import torch

from cogdl.data import Data, Dataset
from cogdl.utils import download_url, untar
import subprocess
from tqdm import tqdm
from langdetect import detect
import random
import argparse

from . import register_dataset

def make_data(raw, processed, samples_per_class = 1000): #raw: raw data, processed: input

    raw_dir = osp.join(raw, 'l0fos')
    if os.path.exists(processed + '/._SUCCESS'):
        return
    classes = []
    for filename in os.listdir(raw_dir):
        if not filename.endswith('.jsonl'):
            continue
        clz = filename.split('.')[0]
        classes.append(clz)
        data = []
        with open('%s/%s' % (raw_dir, filename)) as f:
            for line in tqdm(f, desc='processing %s' % filename):
                obj = json.loads(line.strip())
                try:
                    if detect(obj['title']) != 'en':
                        continue
                    if detect(''.join(obj['abstracts'])) != 'en':
                        continue
                except Exception as e:
                    print('Error: %s, Obj: %s' % (e, obj))
                    continue
                data.append(line)
                if len(data) >= samples_per_class:
                    break
        random.shuffle(data)
        if len(data) < samples_per_class:
            continue
        with open('%s/%s.jsonl' % (processed, clz), 'w') as fout:
            for row in data[:samples_per_class]:
                fout.write('%s' % row)
    with open(processed + '/._SUCCESS', 'w') as f:
        for clz in classes:
            f.write('%s\n' % clz)

@register_dataset("l0fos")
class l0fos(Dataset):

    # def add_args(parser: argparse.ArgumentParser):
    #     parser.add_argument('--samples_per_class', type=int, default=1000)

    def __init__(self):
        self.url = 'https://cloud.tsinghua.edu.cn/f/cd6e3f3276c14e73a9f7/?dl=1'
        dataset = 'l0fos'
        path = osp.join("data", dataset)
        # self.samples_per_class = args.samples_per_class
        super(l0fos, self).__init__(path, dataset)

    def download(self):
        download_url(self.url, self.raw_dir, name='l0fos.zip')
        untar(self.raw_dir, 'l0fos.zip')
        print(f'downloaded to {self.raw_dir}')
    
    def process(self):
        make_data(self.raw_dir, self.processed_dir)

    @property
    def raw_file_names(self):
        ls = [f'l0fos/{i}._SUCCESS' for i in range(0,256)]
        return ls

    def __len__(self):
        return 255

    @property
    def processed_file_names(self):
        return ['art.jsonl','computer science.jsonl','geography.jsonl','mathematics.jsonl', 
        'political science.jsonl', 'biology.jsonl','economics.jsonl','geology.jsonl','medicine.jsonl',
        'psychology.jsonl','business.jsonl','engineering.jsonl','history.jsonl','philosophy.jsonl',
        'sociology.jsonl', 'chemistry.jsonl','environmental science.jsonl','materials science.jsonl','physics.jsonl']