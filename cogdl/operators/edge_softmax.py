import os
import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(__file__))

# SPMM
if not torch.cuda.is_available():
    edge_softmax = None
else:
    try:
        edge_softmax = load(
            name="edge_softmax",
            sources=[os.path.join(path, "edge_softmax/edge_softmax.cc"), os.path.join(path, "edge_softmax/edge_softmax.cu")],
            verbose=False,
        )
        def csr_edge_softmax(rowptr, h):
            return EdgeSoftmaxFunction.apply(rowptr, h)

    except Exception:
        edge_softmax = None


class EdgeSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, h):
        out = edge_softmax.edge_softmax(rowptr, h)
        ctx.backward_csc = (rowptr, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rowptr, out = ctx.backward_csc
        grad_out = grad_out.contiguous()
        grad_softmax =  edge_softmax.edge_softmax_backward(rowptr, out, grad_out)
        return None, grad_softmax
