import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(__file__))

# SPMM

try:
    spmm_max = load(
        name="scatter_max",
        sources=[os.path.join(path, "scatter_max/scatter_max.cc"), os.path.join(path, "scatter_max/scatter_max.cu")],
        verbose=True,
    )

    def scatter_max(rowptr, colind, feat):
        return ScatterMaxFunction.apply(rowptr, colind, feat)


except Exception:
    spmm_max = None


class ScatterMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, feat):
        out, max_id = spmm_max.scatter_max_fp(rowptr, colind, feat)
        ctx.save_for_backward(max_id)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        max_id = ctx.saved_tensors[0]
        out = spmm_max.scatter_max_bp(grad, max_id)
        return None, None, out
