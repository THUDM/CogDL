from .data import Graph, Adjacency
from .batch import Batch, batch_graphs


from cogdl.backend import BACKEND
if BACKEND == 'jittor':
    from .dataset_jt import Dataset, MultiGraphDataset
    from .dataset_jt import Dataset as DataLoader 
elif BACKEND == 'torch':
    from .dataset import Dataset, MultiGraphDataset
    from .dataloader import DataLoader
else:
    raise ("Unsupported backend:", BACKEND)


__all__ = ["Graph", "Adjacency", "Batch", "Dataset", "DataLoader", "MultiGraphDataset", "batch_graphs"]
