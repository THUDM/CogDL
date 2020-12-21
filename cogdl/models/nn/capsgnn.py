import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import GCNConv
import pandas as pd
import random
from tqdm import tqdm, trange
from texttable import Texttable
# import glob
# import json

from cogdl.data import DataLoader
from .. import register_model
from ..base_model import BaseModel
from cogdl.trainers.base_trainer import BaseTrainer

"""Utils."""

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value:i for i, value in enumerate(node_properties)}


"""CapsGNN layers."""

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class PrimaryCapsuleLayer(torch.nn.Module):
    """
    Primary Convolutional Capsule Layer class based on:
    https://github.com/timomernick/pytorch-capsule.
    """
    def __init__(self, in_units, in_channels, num_units, capsule_dimensions):
        super(PrimaryCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.num_units = num_units
        self.units = []
        for i in range(self.num_units):
            unit = torch.nn.Conv1d(in_channels=in_channels,
                                   out_channels=capsule_dimensions,
                                   kernel_size=(in_units, 1),
                                   stride=1,
                                   bias=True)

            self.add_module("unit_" + str(i), unit)
            self.units.append(unit)

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Primary capsule features.
        """
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)
        return PrimaryCapsuleLayer.squash(u)

class SecondaryCapsuleLayer(torch.nn.Module):
    """
    Secondary Convolutional Capsule Layer class based on this repostory:
    https://github.com/timomernick/pytorch-capsule
    """
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(SecondaryCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.W = torch.nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Capsule output.
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).to(u_hat.device)

        num_iterations = 3

        for _ in range(num_iterations):
            c_ij = torch.nn.functional.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = SecondaryCapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)

class Attention(torch.nn.Module):
    """
    2 Layer Attention Module.
    See the CapsGNN paper for details.
    """
    def __init__(self, attention_size_1, attention_size_2):
        super(Attention, self).__init__()
        """
        :param attention_size_1: Number of neurons in 1st attention layer.
        :param attention_size_2: Number of neurons in 2nd attention layer.
        """
        self.attention_1 = torch.nn.Linear(attention_size_1, attention_size_2)
        self.attention_2 = torch.nn.Linear(attention_size_2, attention_size_1)

    def forward(self, x_in):
        """
        Forward propagation pass.
        :param x_in: Primary capsule output.
        :param condensed_x: Attention normalized capsule output.
        """
        attention_score_base = self.attention_1(x_in)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = self.attention_2(attention_score_base)
        attention_score = torch.nn.functional.softmax(attention_score, dim=0)
        condensed_x = x_in *attention_score
        return condensed_x

def margin_loss(scores, target, loss_lambda):
    """
    The margin loss from the original paper. Based on:
    https://github.com/timomernick/pytorch-capsule
    :param scores: Capsule scores.
    :param target: Target groundtruth.
    :param loss_lambda: Regularization parameter.
    :return L_c: Classification loss.
    """
    scores = scores.squeeze()
    v_mag = torch.sqrt((scores**2).sum(dim=1, keepdim=True))
    zero = Variable(torch.zeros(1)).to(target.device)
    m_plus = 0.9
    m_minus = 0.1
    max_l = torch.max(m_plus - v_mag, zero).view(1, -1)**2
    max_r = torch.max(v_mag - m_minus, zero).view(1, -1)**2
    # T_c = Variable(torch.zeros(v_mag.shape)).to(target.device)
    T_c = target
    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c

"""CapsGNN Model."""

@register_model("capsgnn")
class CapsGNN(BaseModel):
    """
    An implementation of themodel described in the following paper:
    https://openreview.net/forum?id=Byl8BnRcYm
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--gcn_filters",
                            type=int, default=20)
        parser.add_argument("--gcn_layers",
                            type=int, default=2)
        parser.add_argument("--capsule_dimensions",
                            type=int, default=8)
        parser.add_argument("--inner_attention_dimensions",
                            type=int, default=20)
        parser.add_argument("--number_of_capsules",
                            type=int, default=8)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.gcn_filters,
            args.gcn_layers,
            args.capsule_dimensions,
            args.inner_attention_dimensions,
            args.number_of_capsules,
            args.num_features,
            args.num_classes,
            args.theta,
            args.lambd
        )

    def __init__(self, gcn_filters, gcn_layers,
                 capsule_dimensions, inner_attention_dimensions,
                 number_of_capsules, number_of_features, number_of_targets, theta, lambd):
        super(CapsGNN, self).__init__()
        self.gcn_filters = gcn_filters
        self.gcn_layers = gcn_layers
        self.capsule_dimensions = capsule_dimensions
        self.inner_attention_dimensions = inner_attention_dimensions
        self.number_of_capsules = number_of_capsules
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self.theta = theta
        self.lambd = lambd
        self._setup_layers()

    def _setup_base_layers(self):
        """
        Creating GCN layers.
        """
        self.base_layers = [GCNConv(self.number_of_features, self.gcn_filters)]
        for _ in range(self.gcn_layers-1):
            self.base_layers.append(GCNConv(self.gcn_filters, self.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        """
        Creating primary capsules.
        """
        self.first_capsule = PrimaryCapsuleLayer(in_units=self.gcn_filters,
                                                 in_channels=self.gcn_layers,
                                                 num_units=self.gcn_layers,
                                                 capsule_dimensions=self.capsule_dimensions)

    def _setup_attention(self):
        """
        Creating attention layer.
        """
        self.attention = Attention(self.gcn_layers*self.capsule_dimensions,
                                   self.inner_attention_dimensions)

    def _setup_graph_capsules(self):
        """
        Creating graph capsules.
        """
        self.graph_capsule = SecondaryCapsuleLayer(self.gcn_layers,
                                                   self.capsule_dimensions,
                                                   self.number_of_capsules,
                                                   self.capsule_dimensions)

    def _setup_class_capsule(self):
        """
        Creating class capsules.
        """
        self.class_capsule = SecondaryCapsuleLayer(self.capsule_dimensions,
                                                   self.number_of_capsules,
                                                   self.number_of_targets,
                                                   self.capsule_dimensions)

    def _setup_reconstruction_layers(self):
        """
        Creating histogram reconstruction layers.
        """
        self.reconstruction_layer_1 = torch.nn.Linear(self.number_of_targets*self.capsule_dimensions,
                                                      int((self.number_of_features*2)/3))

        self.reconstruction_layer_2 = torch.nn.Linear(int((self.number_of_features*2)/3),
                                                      int((self.number_of_features*3)/2))

        self.reconstruction_layer_3 = torch.nn.Linear(int((self.number_of_features*3)/2),
                                                      self.number_of_features)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. GCN layers.
        2. Primary capsules.
        3. Attention
        4. Graph capsules.
        5. Class capsules.
        6. Reconstruction layers.
        """
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()

    def calculate_reconstruction_loss(self, capsule_input, features):
        """
        Calculating the reconstruction loss of the model.
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """

        v_mag = torch.sqrt((capsule_input**2).sum(dim=1))
        _, v_max_index = v_mag.max(dim=0)
        v_max_index = v_max_index.data

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size())).to(capsule_input.device)
        capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
        capsule_masked = capsule_masked.view(1, -1)

        feature_counts = features.sum(dim=0)
        feature_counts = feature_counts/feature_counts.sum()

        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(capsule_masked))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.softmax(self.reconstruction_layer_3(reconstruction_output), dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)
        reconstruction_loss = torch.sum((features-reconstruction_output)**2)
        return reconstruction_loss

    def forward(self, data):
        features = data.x
        edges = data.edge_index
        hidden_representations = []

        for layer in self.base_layers:
            features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.gcn_layers, self.gcn_filters, -1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1, self.gcn_layers*self.capsule_dimensions)
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.gcn_layers,
                                                                     self.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.capsule_dimensions,
                                                                  self.number_of_capsules)
        class_capsule_output = self.class_capsule(reshaped_graph_capsule_output)
        class_capsule_output = class_capsule_output.view(-1, self.number_of_targets*self.capsule_dimensions)
        class_capsule_output = torch.mean(class_capsule_output, dim=0).view(1,
                                                                            self.number_of_targets,
                                                                            self.capsule_dimensions)
        recon = class_capsule_output.view(self.number_of_targets, self.capsule_dimensions)
        reconstruction_losses = self.calculate_reconstruction_loss(recon, data.x)
        margin_losses = margin_loss(class_capsule_output, data.target, self.lambd)

        loss = reconstruction_losses * self.theta + margin_losses
        pred = torch.sqrt((class_capsule_output ** 2).sum(dim=2))

        return pred, loss

    @classmethod
    def split_dataset(self, dataset, args):
        def create_target_tensor(target):
            return torch.FloatTensor([0.0 if i != target else 1.0 for i in range(args.num_classes)])

        random.shuffle(dataset)
        for i, data in enumerate(dataset):
            dataset[i].target = create_target_tensor(dataset[i].y.item())
        batch_size = 1

        train_size = int(len(dataset) * args.train_ratio)
        test_size = int(len(dataset) * args.test_ratio)
        train_loader = DataLoader(dataset[:train_size], batch_size=batch_size)
        test_loader = DataLoader(dataset[-test_size:], batch_size=batch_size)
        if args.train_ratio + args.test_ratio < 1:
            valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=batch_size)
        else:
            valid_loader = test_loader

        return train_loader, valid_loader, test_loader