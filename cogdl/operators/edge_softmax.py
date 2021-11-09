import os
import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(__file__))

try:
    edge_softmax = load(
        name="edge_softmax",
        sources=[
            os.path.join(path, "edge_softmax/edge_softmax.cc"),
            os.path.join(path, "edge_softmax/edge_softmax.cu"),
        ],
        verbose=False,
    )

    def csr_edge_softmax(rowptr, h):
        return EdgeSoftmaxFunction.apply(rowptr, h)


except Exception:
    edge_softmax = None
    csr_edge_softmax = None


class EdgeSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, h):
        out = edge_softmax.edge_softmax(rowptr, h)
        ctx.save_for_backward(rowptr, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rowptr, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_softmax = edge_softmax.edge_softmax_backward(rowptr, out, grad_out)
        return None, grad_softmax
