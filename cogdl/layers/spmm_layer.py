import os
import torch
from torch.utils.cpp_extension import load

path = os.path.realpath(__file__)
path = os.path.join(os.path.dirname(path), "../operators")

if not torch.cuda.is_available():
    spmm = None
else:
    try:
        spmm = load(
            name="spmm", sources=[os.path.join(path, "spmm.cpp"), os.path.join(path, "spmm_kernel.cu")], verbose=False
        )

        def csrspmm(rowptr, colind, colptr, rowind, x, csr_data, csc_data):
            return SPMMFunction.apply(rowptr, colind, colptr, rowind, x, csr_data, csc_data)

    except Exception as e:
        print(e)
        spmm = None


class SPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, feat, edge_weight_csr=None, edge_weight_csc=None):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)

        ctx.backward_csc = (colptr, rowind, feat, edge_weight_csr, edge_weight_csc)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        colptr, rowind, feat, edge_weight_csr, edge_weight_csc = ctx.backward_csc
        if edge_weight_csr is not None:
            if edge_weight_csc is None:
                raise RuntimeError(
                    "Backward of SPMM require edge values in both src-first and dst-first order, \
                    and do not support gradients for edge values. \
                        Call with SPMMFunction.apply(rowptr, colind, colptr, rowind, in_feat, edge_value_row_first, edge_value_col_first"
                )
            else:
                grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
                grad_edge_weight = None
                # print("[I] Treat edge weight as no_grad.")
        else:
            grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            grad_edge_weight = None

        return None, None, None, None, grad_feat, grad_edge_weight, None
