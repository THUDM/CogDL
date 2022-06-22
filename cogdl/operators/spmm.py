import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

path = os.path.join(os.path.dirname(__file__))

# SPMM


try:
    spmm = load(
        name="spmm",
        extra_ldflags=["-lcusparse"],
        sources=[os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")],
        verbose=False,
    )
    sddmm = load(
        name="sddmm",
        sources=[os.path.join(path, "spmm/sddmm.cpp"), os.path.join(path, "spmm/sddmm_kernel.cu")],
        verbose=False,
    )

    def csrspmm(rowptr, colind, x, csr_data, sym=False, actnn=False):
        if actnn:
            return ActSPMMFunction.apply(rowptr, colind, x, csr_data, sym)
        return SPMMFunction.apply(rowptr, colind, x, csr_data, sym)


except Exception:
    csrspmm = None


try:
    spmm_cpu = load(
        name="spmm_cpu", extra_cflags=["-fopenmp"], sources=[os.path.join(path, "spmm/spmm_cpu.cpp")], verbose=False,
    )
    spmm_cpu = spmm_cpu.csr_spmm_cpu
except Exception:
    spmm_cpu = None


class SPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            ctx.backward_csc = (rowptr, colind, feat, edge_weight_csr, sym)
        else:
            ctx.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if len(ctx.backward_csc) == 5:
            rowptr, colind, feat, edge_weight_csr, sym = ctx.backward_csc
        else:
            rowptr, colind, edge_weight_csr, sym = ctx.backward_csc
        if edge_weight_csr is not None:
            grad_out = grad_out.contiguous()
            if sym:
                colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
            else:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
            grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
            if edge_weight_csr.requires_grad:
                grad_edge_weight = sddmm.csr_sddmm(rowptr, colind, grad_out, feat)
            else:
                grad_edge_weight = None
        else:
            if sym is False:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
                grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            else:
                grad_feat = spmm.csr_spmm_no_edge_value(rowptr, colind, grad_out)
            grad_edge_weight = None
        return None, None, grad_feat, grad_edge_weight, None


try:
    from actnn.ops import quantize_activation, dequantize_activation
except Exception:
    pass


class ActSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            quantized = quantize_activation(feat, None)
            ctx.backward_csc = (rowptr, colind, quantized, edge_weight_csr, sym)
            ctx.other_args = feat.shape
        else:
            ctx.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if len(ctx.backward_csc) == 5:
            rowptr, colind, quantized, edge_weight_csr, sym = ctx.backward_csc
            q_input_shape = ctx.other_args
            feat = dequantize_activation(quantized, q_input_shape)
            del quantized
        else:
            rowptr, colind, edge_weight_csr, sym = ctx.backward_csc
        del ctx.backward_csc

        if edge_weight_csr is not None:
            grad_out = grad_out.contiguous()
            if sym:
                colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
            else:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
            grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
            if edge_weight_csr.requires_grad:
                grad_edge_weight = sddmm.csr_sddmm(rowptr, colind, grad_out, feat)
            else:
                grad_edge_weight = None
        else:
            if sym is False:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
                grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            else:
                grad_feat = spmm.csr_spmm_no_edge_value(rowptr, colind, grad_out)
            grad_edge_weight = None
        return None, None, grad_feat, grad_edge_weight, None
