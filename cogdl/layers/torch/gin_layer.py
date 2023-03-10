import torch
import torch.nn as nn

from cogdl.utils import spmm


class GINLayer(nn.Module):
    r"""Graph Isomorphism Network layer from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{sum}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

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
        super(GINLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, graph, x):
        out = (1 + self.eps) * x + spmm(graph, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out
