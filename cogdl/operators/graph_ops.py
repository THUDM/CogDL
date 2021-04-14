import os
import torch
from torch.utils.cpp_extension import load


path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "operators")

# subgraph and sample_adj
try:
    sample = load(name="sampler", sources=[os.path.join(path, "sample/sample.cpp")], verbose=False)
    subgraph_c = sample.subgraph
    sample_adj_c = sample.sample_adj
    coo2csr_cpu = sample.coo2csr_cpu
    coo2csr_cpu_index = sample.coo2csr_cpu_index
except Exception as e:
    print(e)
    subgraph_c = None
    sample_adj_c = None
    coo2csr_cpu_index = None
    coo2csr_cpu = None
