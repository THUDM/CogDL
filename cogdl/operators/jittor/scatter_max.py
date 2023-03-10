import os
import numpy as np
import torch
from jittor.compiler import compile_torch_extensions

path = os.path.join(os.path.dirname(__file__))

# SPMM

try:
    compile_torch_extensions(
        "scatter_max",
        [os.path.join(path, "scatter_max/scatter_max.cc"), os.path.join(path, "scatter_max/scatter_max.cu")],
        [], [], [],1, 1
    )
    import scatter_max as spmm_max

    def scatter_max(rowptr, colind, feat):
        return ScatterMaxFunction.apply(rowptr, colind, feat)


except Exception:
    spmm_max = None


class ScatterMaxFunction(torch.autograd.Function):
    def execute(self, rowptr, colind, feat):
        out, max_id = spmm_max.scatter_max_fp(rowptr, colind, feat)
        self.backward_spmm_max = (max_id)
        return out

    def grad(self, grad):
        grad = grad.clone()
        max_id = self.backward_spmm_max[0]
        out = spmm_max.scatter_max_bp(grad, max_id)
        return None, None, out
