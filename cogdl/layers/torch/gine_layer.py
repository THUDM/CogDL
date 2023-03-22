import torch
import torch.nn.functional as F
from cogdl.utils import spmm

from . import BaseLayer


class GINELayer(BaseLayer):
    r"""The modified GINConv operator from the `"Graph convolutions that can finally model local structure" paper
     <https://arxiv.org/pdf/2011.15069.pdf>`__.

    Parameters
    ----------
    apply_func : callable layer function)
        layer or function applied to update node feature
    eps : float32, optional
        Initial `\epsilon` value.
    train_eps : bool, optional
        If True, `\epsilon` will be a learnable parameter.
    """

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINELayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, graph, x):
        # m = self.message(x[graph.edge_index[0]], graph.edge_attr)
        # out = self.aggregate(graph, m)
        out = spmm(graph, x)
        out += (1 + self.eps) * x
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out

    def message(self, x, attr):
        return F.relu(x + attr)
