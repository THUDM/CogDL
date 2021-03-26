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


# SPMM
if not torch.cuda.is_available():
    spmm = None
else:
    try:
        spmm = load(
            name="spmm",
            sources=[os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")],
            verbose=False,
        )
        sddmm = load(
            name="sddmm",
            sources=[os.path.join(path, "spmm/sddmm.cpp"), os.path.join(path, "spmm/sddmm_kernel.cu")],
            verbose=False,
        )

        def csrspmm(rowptr, colind, colptr, rowind, x, csr_data, sym=False):
            return SPMMFunction.apply(rowptr, colind, colptr, rowind, x, csr_data, sym)

    except Exception as e:
        print(e)
        spmm = None


class SPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        ctx.backward_csc = (rowptr, colind, colptr, rowind, feat, edge_weight_csr, sym)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rowptr, colind, colptr, rowind, feat, edge_weight_csr, sym = ctx.backward_csc
        if edge_weight_csr is not None:
            grad_out = grad_out.contiguous()
            edge_weight_csc = spmm.csr2csc(rowptr, colind, colptr, rowind, edge_weight_csr)
            grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
            grad_edge_weight = sddmm.csr_sddmm(rowptr, colind, grad_out, feat)
            # TODO: symmetry
        else:
            grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            grad_edge_weight = None
        return None, None, None, None, grad_feat, grad_edge_weight, None
