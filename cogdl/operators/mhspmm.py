import os

import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(__file__))

# SPMM
try:
    mhspmm = load(
        name="mhspmm",
        sources=[os.path.join(path, "spmm/multiheadSpmm.cpp"), os.path.join(path, "spmm/multiheadSpmm.cu")],
        verbose=False,
    )
    mhsddmm = load(
        name="mhsddmm",
        sources=[os.path.join(path, "spmm/multiheadSddmm.cpp"), os.path.join(path, "spmm/multiheadSddmm.cu")],
        verbose=False,
    )
    mhtranspose = load(
        name="mhtranspose",
        extra_ldflags=["-lcusparse"],
        sources=[os.path.join(path, "spmm/mhTranspose.cpp"), os.path.join(path, "spmm/mhTranspose.cu")],
        verbose=False,
    )

    spmm = load(
        name="spmm",
        extra_ldflags=["-lcusparse"],
        sources=[os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")],
        verbose=False,
    )

    def csrmhspmm(rowptr, colind, feat, attention):
        return MHSPMMFunction.apply(rowptr, colind, feat, attention)


except Exception:
    mhspmm = None
    csrmhspmm = None
    spmm = None


class MHSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, feat, attention):
        out = mhspmm.mhspmm(rowptr, colind, attention, feat)
        ctx.save_for_backward(rowptr, colind, feat, attention)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rowptr, colind, feat, attention = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        numlist = torch.arange(colind.size(0), device=grad_out.device, dtype=torch.int32)
        # colptr, rowind, permute = mhtranspose.csr2csc(rowptr, colind, numlist)

        colptr, rowind, permute = spmm.csr2csc(rowptr, colind, numlist.float())
        permute = permute.int()
        attention_csc = mhtranspose.mhtranspose(permute, attention)
        grad_feat = mhspmm.mhspmm(colptr, rowind, attention_csc, grad_out)
        grad_edge_weight = mhsddmm.mhsddmm(rowptr, colind, grad_out, feat)

        return None, None, grad_feat, grad_edge_weight
