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

from .. import register_model
from ..base_model import BaseModel

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
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))

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
    zero = Variable(torch.zeros(1))
    m_plus = 0.9
    m_minus = 0.1
    max_l = torch.max(m_plus - v_mag, zero).view(1, -1)**2
    max_r = torch.max(v_mag - m_minus, zero).view(1, -1)**2
    T_c = Variable(torch.zeros(v_mag.shape))
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
        parser.add_argument("--number_of_features",
                            type=int)
        parser.add_argument("--number_of_targets",
                            type=int)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.gcn_filters,
            args.gcn_layers,
            args.capsule_dimensions,
            args.inner_attention_dimensions,
            args.number_of_capsules,
            args.number_of_features,
            args.number_of_targets
        )

    def __init__(self, gcn_filters, gcn_layers,
                 capsule_dimensions, inner_attention_dimensions,
                 number_of_capsules, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        self.gcn_filters = gcn_filters
        self.gcn_layers = gcn_layers
        self.capsule_dimensions = capsule_dimensions
        self.inner_attention_dimensions = inner_attention_dimensions
        self.number_of_capsules = number_of_capsules
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
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

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
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
        """
        Forward propagation pass.
        :param data: Dictionary of tensors with features and edges.
        :return class_capsule_output: Class capsule outputs.
        """
        features = data["features"]
        edges = data["edges"]
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
        reconstruction_loss = self.calculate_reconstruction_loss(recon, data["features"])
        return class_capsule_output, reconstruction_loss

"""CapsGNN Trainer."""

class CapsGNNTrainer(object):
    """
    CapsGNN training and scoring.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets in order to setup weights later.
        """
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)
        graph_paths = self.train_graph_paths + self.test_graph_paths

        targets = set()
        features = set()
        for path in tqdm(graph_paths):
            data = json.load(open(path))
            targets = targets.union(set([data["target"]]))
            features = features.union(set(data["labels"]))

        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = len(self.feature_map)
        self.number_of_targets = len(self.target_map)

    def setup_model(self):
        """
        Enumerating labels and initializing a CapsGNN.
        """
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets)

    def create_batches(self):
        """
        Batching the graphs for training.
        """
        self.batches = []
        for i in range(0, len(self.train_graph_paths), self.args.batch_size):
            self.batches.append(self.train_graph_paths[i:i+self.args.batch_size])

    def create_data_dictionary(self, target, edges, features):
        """
        Creating a data dictionary.
        :param target: Target vector.
        :param edges: Edge list tensor.
        :param features: Feature tensor.
        """
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return : Target vector.
        """
        return  torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(self.number_of_targets)])

    def create_edges(self, data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        edges = [[edge[0], edge[1]] for edge in data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    def create_features(self, data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        features = np.zeros((len(data["labels"]), self.number_of_features))
        node_indices = [node for node in range(len(data["labels"]))]
        feature_indices = [self.feature_map[label] for label in data["labels"].values()]
        features[node_indices, feature_indices] = 1.0
        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path):
        """
        Creating tensors and a data dictionary with Torch tensors.
        :param path: path to the data JSON.
        :return to_pass_forward: Data dictionary.
        """
        data = json.load(open(path))
        target = self.create_target(data)
        edges = self.create_edges(data)
        features = self.create_features(data)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    def fit(self):
        """
        Training a model on the training set.
        """
        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        for _ in tqdm(range(self.args.epochs), desc="Epochs: ", leave=True):
            random.shuffle(self.train_graph_paths)
            self.create_batches()
            losses = 0
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    data = self.create_input_data(path)
                    prediction, reconstruction_loss = self.model(data)
                    loss = margin_loss(prediction,
                                       data["target"],
                                       self.args.lambd)
                    loss = loss+self.args.theta*reconstruction_loss
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                average_loss = losses/(step + 1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss, 4))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nScoring.\n")
        self.model.eval()
        self.predictions = []
        self.hits = []
        for path in tqdm(self.test_graph_paths):
            data = self.create_input_data(path)
            prediction, _ = self.model(data)
            prediction_mag = torch.sqrt((prediction**2).sum(dim=2))
            _, prediction_max_index = prediction_mag.max(dim=1)
            prediction = prediction_max_index.data.view(-1).item()
            self.predictions.append(prediction)
            self.hits.append(data["target"][prediction] == 1.0)

        print("\nAccuracy: " + str(round(np.mean(self.hits), 4)))

    def save_predictions(self):
        """
        Saving the test set predictions.
        """
        identifiers = [path.split("/")[-1].strip(".json") for path in self.test_graph_paths]
        out = pd.DataFrame()
        out["id"] = identifiers
        out["predictions"] = self.predictions
        out.to_csv(self.args.prediction_path, index=None)
