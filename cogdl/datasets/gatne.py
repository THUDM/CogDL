import os.path as osp

import torch

from cogdl.data import Dataset, download_url
from cogdl.read import read_gatne_data

from . import register_dataset


class GatneDataset(Dataset):
    r"""The network datasets "Amazon", "Twitter" and "YouTube" from the
    `"Representation Learning for Attributed Multiplex Heterogeneous Network"
    <https://arxiv.org/abs/1905.01669>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Amazon"`,
            :obj:`"Twitter"`, :obj:`"YouTube"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`cogdl.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`cogdl.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/THUDM/GATNE/raw/master/data'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(GatneDataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['train.txt', 'valid.txt', 'test.txt']
        return ['{}/{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_gatne_data(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


@register_dataset('amazon')
class AmazonDataset(GatneDataset):
    def __init__(self):
        dataset = 'amazon'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(AmazonDataset, self).__init__(path, dataset)


@register_dataset('twitter')
class TwitterDataset(GatneDataset):
    def __init__(self):
        dataset = 'twitter'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(TwitterDataset, self).__init__(path, dataset)


@register_dataset('youtube')
class YouTubeDataset(GatneDataset):
    def __init__(self):
        dataset = 'youtube'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(YouTubeDataset, self).__init__(path, dataset)
