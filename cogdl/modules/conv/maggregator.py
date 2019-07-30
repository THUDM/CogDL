import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from cogdl.modules.conv import MessagePassing
from cogdl.utils import remove_self_loops, add_self_loops

from ..inits import glorot, zeros


class meanaggr(torch.nn.Module):
   
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(meanaggr, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(x,edge_index,  num_nodes):
        # here edge_index is already a sparse tensor
        deg = torch.sparse.sum(edge_index,1)
        deg_inv= deg.pow(-1).to_dense()
        
        x=torch.matmul(edge_index,x)
        #  print(x,deg_inv)
        x=x.t()*deg_inv
        #  x：512*dim, edge_weight：256*512

        return x.t()

    def forward(self, x, edge_index, num_nodes,edge_weight=None,bias=True):
        """"""
        x = torch.matmul(x, self.weight)
        if bias:
            x=x+self.bias

        x = self.norm(x,edge_index,num_nodes)
        
        return x

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
