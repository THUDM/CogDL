import torch
import torch.nn as nn
import torch.nn.functional as F

from graphmae.utils import create_activation, NormLayer, create_norm
from cogdl.layers import GINLayer


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(
                GINLayer(apply_func, eps=0, train_eps=learn_eps)
            )
            if last_residual:
                if in_dim != out_dim:
                    self.residual_layers.append(nn.Linear(in_dim, out_dim, bias=False))
                else:
                    self.residual_layers.append(nn.Identity())
        else:
            # input projection (no residual)
            self.layers.append(GINLayer(
                    ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    eps=0,
                    train_eps=learn_eps,
                )
            )
            if residual:
                if in_dim != num_hidden:
                    self.residual_layers.append(nn.Linear(in_dim, num_hidden, bias=False))
                else:
                    self.residual_layers.append(nn.Identity())

            for l in range(1, num_layers - 1):
                self.layers.append(
                    GINLayer(
                        ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                        eps=0,
                        train_eps=learn_eps,
                    )
                )
                if residual:
                    self.residual_layers.append(nn.Identity())
            # output projection
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(
                GINLayer(
                    apply_func, eps=0, train_eps=learn_eps
                )
            )
            if last_residual:
                if num_hidden != out_dim:
                    self.residual_layers.append(nn.Linear(num_hidden, out_dim, bias=False))
                else:
                    self.residual_layers.append(nn.Identity())

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        pre_h = inputs
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            if self.residual:
                h = self.residual[l](pre_h) + h
            pre_h = h
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.norm = create_norm(norm)(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)