import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add, scatter
from .. import BaseModel, register_model


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)
        
    def forward(self, x):
        output = torch.mm(x, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


@register_model("grand")
class Grand(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--hidden_dropout", type=float, default=0.5)
        parser.add_argument("--input_dropout", type=float, default=0.5)
        parser.add_argument("--bn", type=bool, default=False)
        parser.add_argument("--dropnode_rate", type=float, default=0.5)
        parser.add_argument('--order', type = int, default = 5)
        parser.add_argument('--tem', type = float, default = 0.5)
        parser.add_argument('--lam', type = float, default = 0.5)
        parser.add_argument('--sample', type = int, default = 2)    
        parser.add_argument('--alpha', type = float, default = 0.2)        
    


        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.input_dropout, args.hidden_dropout, args.bn, args.dropnode_rate, args.tem, args.lam, args.order, args.sample, args.alpha)

    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn, dropnode_rate, tem, lam, order, sample, alpha):
        super(Grand, self).__init__()
        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.tem = tem
        self.lam = lam
        self.order = order
        self.dropnode_rate = dropnode_rate
        self.sample = sample
        self.alpha = alpha
    
    def dropNode(self, x):
        n = x.shape[0]
        drop_rates = torch.ones(n) * self.dropnode_rate
        if self.training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x

        else:
            x =  x * (1. - self.dropnode_rate)
        return x

    def normalize_adj(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim = 0, dim_size = num_nodes)
        
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        #print(edge_weight)
        return edge_weight

    def rand_prop(self, x, edge_index, edge_weight):
        edge_weight = self.normalize_adj(edge_index, edge_weight, x.shape[0])
        row, col = edge_index[0], edge_index[1]
        x = self.dropNode(x)

        y = x
        for i in range(self.order):
            x_source = x[col]
            x = scatter(x_source * edge_weight[:, None], row[:,None], dim=0, dim_size=x.shape[0], reduce='sum').detach_()
            #x = torch.spmm(adj, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()
    
    def consis_loss(self, logps, train_mask):
        temp = self.tem
        ps = [torch.exp(p)[~train_mask] for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))
        loss = loss/len(ps)

        return self.lam * loss
    
    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow_(-1)
        row_inv.masked_fill_(row_inv == float('inf'), 0)
        x = x * row_inv[:, None]
        return x

    def forward(self, x, edge_index):
        """
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]).float(),
            (x.shape[0], x.shape[0]),
        ).to(x.device)
        """
        x = self.normalize_x(x)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32).to(x.device)
        x = self.rand_prop(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn1(x) 
        x = F.dropout(x, self.input_droprate, training = self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training = self.training)
        x = self.layer2(x)

        return F.log_softmax(x, dim=-1)
    
    def loss(self, data):
        output_list = []
        for i in range(self.sample):
            output_list.append(self.forward(data.x, data.edge_index))
        loss_train = 0.
        for output in output_list:
            loss_train += F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss_train = loss_train/self.sample
        loss_consis = self.consis_loss(output_list, data.train_mask)
        
        return loss_train + loss_consis

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
