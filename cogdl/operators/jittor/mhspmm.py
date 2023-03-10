import os
import jittor
from jittor.compiler import compile_torch_extensions
from jittor import Function
path = os.path.join(os.path.dirname(__file__))

# SPMM
try:
    compile_torch_extensions("mhspmm",
        [os.path.join(path, "spmm/multiheadSpmm.cpp"), os.path.join(path, "spmm/multiheadSpmm.cu")],[], [], [],1, 1

    )
    compile_torch_extensions("mhsddmm",
        [os.path.join(path, "spmm/multiheadSddmm.cpp"), os.path.join(path, "spmm/multiheadSddmm.cu")],[], [], [],1, 1
    )
    compile_torch_extensions("mhtranspose",[os.path.join(path, "spmm/mhTranspose.cpp"), os.path.join(path, "spmm/mhTranspose.cu")],[], [], [],1, 1
    )

    compile_torch_extensions("spmm",
        [os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")],[], [], [],1, 1
    )
    import spmm
    import mhspmm
    import mhsddmm
    import mhtranspose
    def csrmhspmm(rowptr, colind, feat, attention):
        return MHSPMMFunction.apply(rowptr, colind, feat, attention)


except Exception:
    mhspmm = None
    csrmhspmm = None
    spmm = None


class MHSPMMFunction(Function):
    def execute(self, rowptr, colind, feat, attention):
        out = mhspmm.mhspmm(rowptr, colind, attention, feat)
        self.backward_mhspmm=(rowptr, colind, feat, attention)

        return out

    def grad(self, grad_out):
        rowptr, colind, feat, attention = self.backward_mhspmm
        numlist = jittor.arange(colind.size(0), dtype=jittor.int32)
        # colptr, rowind, permute = mhtranspose.csr2csc(rowptr, colind, numlist)

        colptr, rowind, permute = spmm.csr2csc(rowptr, colind, numlist.float())
        permute = permute.int()
        attention_csc = mhtranspose.mhtranspose(permute, attention)
        grad_feat = mhspmm.mhspmm(colptr, rowind, attention_csc, grad_out)
        grad_edge_weight = mhsddmm.mhsddmm(rowptr, colind, grad_out, feat)

        return None, None, grad_feat, grad_edge_weight
