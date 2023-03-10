import math

import torch
import torch.nn as nn

from cogdl.utils import get_activation


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / nn.math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)


class PPRGoLayer(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, activation="relu"):
        super(PPRGoLayer, self).__init__()
        self.dropout = dropout
        self.nlayers = num_layers
        shapes = [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList()
        self.layers.append(LinearLayer(in_feats, hidden_size, bias=False))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(shapes[i], shapes[i + 1], bias=False))
        self.activation = get_activation(activation)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = layer(h)
            if i != self.nlayers - 1:
                h = self.activation(h)
        return h
