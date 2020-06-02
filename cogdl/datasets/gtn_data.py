import sys
import time
import os
import os.path as osp
import requests
import shutil
import tqdm
import pickle
import numpy as np

import torch

from cogdl.data import Data, Dataset, download_url

from . import register_dataset

def download(url, path, fname, redownload=False):
    """
    Downloads file using `requests`. If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``True``).
    """
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload
    print("[ downloading: " + url + " to " + outfile + " ]")
    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(fname))

    while download and retry >= 0:
        resume_file = outfile + '.part'
        resume = os.path.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = 'ab'
        else:
            resume_pos = 0
            mode = 'wb'
        response = None

        with requests.Session() as session:
            try:
                header = {'Range': 'bytes=%d-' % resume_pos,
                          'Accept-Encoding': 'identity'} if resume else {}
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get('Accept-Ranges', 'none') == 'none':
                    resume_pos = 0
                    mode = 'wb'

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos

                with open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except requests.exceptions.ConnectionError:
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print('Connection error, retrying. (%d retries left)' % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning('Received less data than specified in ' +
                                 'Content-Length header for ' + url + '.' +
                                 ' There may be a download problem.')
        move(resume_file, outfile)

    pbar.close()


def move(path1, path2):
    """Renames the given file."""
    shutil.move(path1, path2)


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)

class GTNDataset(Dataset):
    r"""The network datasets "ACM", "DBLP" and "IMDB" from the
    `"Graph Transformer Networks"
    <https://arxiv.org/abs/1911.06455>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Amazon"`,
            :obj:`"Twitter"`, :obj:`"YouTube"`).
    """

    urls = {'gtn-acm': 'https://www.dropbox.com/s/x5suqwasrji7u0l/gtn-acm.zip?dl=1',
           'gtn-dblp': 'https://www.dropbox.com/s/jcta41n3cfic3d0/gtn-dblp.zip?dl=1',
           'gtn-imdb': 'https://www.dropbox.com/s/om3r1w9y5y43t9g/gtn-imdb.zip?dl=1'}

    def __init__(self, root, name):
        self.name = name
        self.url = self.urls[name]
        super(GTNDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])
        self.num_classes = torch.max(self.data.train_target).item() + 1
        self.num_edge = len(self.data.adj)
        self.num_nodes = self.data.x.shape[0]

    @property
    def raw_file_names(self):
        names = ["edges.pkl", "labels.pkl", "node_features.pkl"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def read_gtn_data(self, folder):
        edges = pickle.load(open(osp.join(folder, 'edges.pkl'), 'rb'))
        labels = pickle.load(open(osp.join(folder, 'labels.pkl'), 'rb'))
        node_features = pickle.load(open(osp.join(folder, 'node_features.pkl'), 'rb'))

        data = Data()
        data.x = torch.from_numpy(node_features).type(torch.FloatTensor)

        num_nodes = edges[0].shape[0]

        A = []
        
        for i,edge in enumerate(edges):
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
            A.append((edge_tmp,value_tmp))
        edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
        A.append((edge_tmp,value_tmp))
        data.adj = A

        data.train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
        data.train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
        data.valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
        data.valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
        data.test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
        data.test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

        self.data = data

    def get(self, idx):
        assert idx == 0
        return self.data
    
    def apply_to_device(self, device):
        self.data.x = self.data.x.to(device)

        self.data.train_node = self.data.train_node.to(device)
        self.data.valid_node = self.data.valid_node.to(device)
        self.data.test_node = self.data.test_node.to(device)

        self.data.train_target = self.data.train_target.to(device)
        self.data.valid_target = self.data.valid_target.to(device)
        self.data.test_target = self.data.test_target.to(device)

        new_adj = []
        for (t1, t2) in self.data.adj:
            new_adj.append((t1.to(device), t2.to(device)))
        self.data.adj = new_adj

    def download(self):
        download(self.url, '../../data', 'gtn-data.zip')

    def process(self):
        self.read_gtn_data(self.raw_dir)
        torch.save(self.data, self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


@register_dataset("gtn-acm")
class ACM_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-acm"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ACM_GTNDataset, self).__init__(path, dataset)


@register_dataset("gtn-dblp")
class DBLP_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-dblp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(DBLP_GTNDataset, self).__init__(path, dataset)


@register_dataset("gtn-imdb")
class IMDB_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-imdb"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(IMDB_GTNDataset, self).__init__(path, dataset)
