import os
import jittor as jt

from jittor import Function
from jittor.compiler import compile_torch_extensions

jt.flags.use_cuda = 1
cached_op = {"csr_spmm": None}


def tensor2jit(x):
    return jt.array(x.cpu().numpy())


def init_spmm_ops():
    if cached_op["csr_spmm"] is None:
        op_path = os.path.abspath(__file__)
        spmm_path = os.path.join(os.path.dirname(op_path), "spmm/spmm.cpp")
        spmm_cu_path = os.path.join(os.path.dirname(op_path), "spmm/spmm_kernel.cu")
        compile_torch_extensions("spmm", [spmm_path, spmm_cu_path], 1, 1)
        from spmm import csr_spmm

        cached_op["csr_spmm"] = csr_spmm


def spmm(graph, x):
    row_ptr, col_indices = graph.row_indptr, graph.col_indices
    csr_data = graph.edge_weight
    spmm = SPMM()
    x = spmm(tensor2jit(row_ptr.int()), tensor2jit(col_indices.int()), x, tensor2jit(csr_data))
    return x


class SPMM(Function):
    def execute(self, rowptr, colind, feat, edge_weight_csr=None):
        init_spmm_ops()
        self.csr_spmm = cached_op["csr_spmm"]

        out = self.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        self.backward_csc = (rowptr, colind, edge_weight_csr)
        return out

    def grad(self, grad_out):
        rowptr, colind, edge_weight_csr = self.backward_csc
        colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
        grad_feat = self.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)

        return None, None, grad_feat, None
