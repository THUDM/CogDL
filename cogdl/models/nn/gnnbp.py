import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model

class Dense(nn.Module):
    
    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


@register_model("gnnbp")
class GNNBP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--bias", type=str, default='none')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.bias,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout, bias):
        super(GNNBP, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        
        self.layers = nn.ModuleList()
        self.layers.append(Dense(num_features, hidden_size, bias))
        for i in range(num_layers-2):
            self.layers.append(Dense(hidden_size, hidden_size, bias))
        self.layers.append(Dense(hidden_size, num_classes))
        self.activation = nn.ReLU()


    def forward(self, x, edge_index):
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation(layer(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
    
    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
