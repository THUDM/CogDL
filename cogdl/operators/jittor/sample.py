import os
from jittor.compiler import compile_torch_extensions

path = os.path.join(os.path.dirname(__file__))

try:
    compile_torch_extensions("sampler", os.path.join(path,"sample/sample.cpp"), [], [], [],1, 1)
    import sampler as sample
    subgraph_c = sample.subgraph
    sample_adj_c = sample.sample_adj
    coo2csr_cpu = sample.coo2csr_cpu
    coo2csr_cpu_index = sample.coo2csr_cpu_index
except Exception:
    subgraph_c = None
    sample_adj_c = None
    coo2csr_cpu_index = None
    coo2csr_cpu = None
