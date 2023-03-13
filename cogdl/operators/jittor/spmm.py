import os
import numpy as np
import jittor
from jittor import Function
from jittor.compiler import compile_torch_extensions

path = os.path.join(os.path.dirname(__file__))

# SPMM

try:
    compile_torch_extensions(
        "spmm", [os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")], [], [], [], 1, 1
    )
    compile_torch_extensions(
        "sddmm", [os.path.join(path, "spmm/sddmm.cpp"), os.path.join(path, "spmm/sddmm_kernel.cu")], [], [], [], 1, 1
    )
    import spmm
    import sddmm

    def csrspmm(rowptr, colind, x, csr_data, sym=False, actnn=False):
        if actnn:
            return ActSPMMFunction.apply(rowptr, colind, x, csr_data, sym)
        return SPMMFunction.apply(rowptr, colind, x, csr_data, sym)

except Exception:
    csrspmm = None


try:
    compile_torch_extensions("spmm_cpu", [os.path.join(path, "spmm/spmm_cpu.cpp")], [], [], [], 1, 1)
    import spmm_cpu

    spmm_cpu = spmm_cpu.csr_spmm_cpu
except Exception:
    spmm_cpu = None


class SPMMFunction(Function):
    def execute(self, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            self.backward_csc = (rowptr, colind, feat, edge_weight_csr, sym)
        else:
            self.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    def grad(self, grad_out):
        if len(self.backward_csc) == 5:
            rowptr, colind, feat, edge_weight_csr, sym = self.backward_csc
        else:
            rowptr, colind, edge_weight_csr, sym = self.backward_csc
        if edge_weight_csr is not None:
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

    # def grad(self, grad_out):
    #     rowptr, colind, edge_weight_csr ,edge_weight_csr ,sym= self.backward_csc
    #     colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
    #     grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)

    #     return None, None, grad_feat, None


try:
    from actnn.ops import quantize_activation, dequantize_activation
except Exception:
    pass


class ActSPMMFunction(Function):
    @staticmethod
    def execute(self, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            quantized = quantize_activation(feat, None)
            self.backward_csc = (rowptr, colind, quantized, edge_weight_csr, sym)
            self.other_args = feat.shape
        else:
            self.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    @staticmethod
    def grad(self, grad_out):
        if len(self.backward_csc) == 5:
            rowptr, colind, quantized, edge_weight_csr, sym = self.backward_csc
            q_input_shape = self.other_args
            feat = dequantize_activation(quantized, q_input_shape)
            del quantized
        else:
            rowptr, colind, edge_weight_csr, sym = self.backward_csc
        del self.backward_csc

        if edge_weight_csr is not None:
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
