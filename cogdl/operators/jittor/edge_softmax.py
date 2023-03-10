import os
from jittor.compiler import compile_torch_extensions
from jittor import Function
path = os.path.join(os.path.dirname(__file__))


try:
    compile_torch_extensions("edge_softmax",[
            os.path.join(path, "edge_softmax/edge_softmax.cc"),
            os.path.join(path, "edge_softmax/edge_softmax.cu"),
        ],[], [], [],1, 1
    )
    import edge_softmax

    def csr_edge_softmax(rowptr, h):
        return EdgeSoftmaxFunction.apply(rowptr, h)


except Exception:
    edge_softmax = None
    csr_edge_softmax = None


class EdgeSoftmaxFunction(Function):
    def execute(self, rowptr, h):
        out = edge_softmax.edge_softmax(rowptr, h)
        self.backward_edge_softmax=(rowptr, h)
        return out

    def grad(self, grad_out):
        rowptr, h = self.backward_edge_softmax
        grad_softmax = edge_softmax.edge_softmax_backward(rowptr, h, grad_out)
        return None, grad_softmax
