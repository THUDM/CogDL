import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GCNLayer

from graphmae.utils import create_activation


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = activation if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GCNLayer(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            self.gcn_layers.append(GCNLayer(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GCNLayer(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            self.gcn_layers.append(GCNLayer(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
