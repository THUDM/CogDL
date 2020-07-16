import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.models.nn.srgcn import SRGCN, SSRGCN

from cogdl.models.automl.search_space import SearchSpace


class GNNNet(nn.Module):
    def __init__(self, actions, args):
        super(GNNNet, self).__init__()
        self.model = None
        search_space = SearchSpace()
        self.search_space = list(search_space.get_search_space().keys())

        self.build_model(actions, args)

    def get_action_index(self, action):
        index = self.search_space.index(action)
        assert index > -1
        return index

    def build_model(self, actions, args, format="one"):
        attention_type = actions[self.get_action_index("attention_type")]
        activation = actions[self.get_action_index("activation")]
        num_heads = actions[self.get_action_index("num_heads")]
        hidden_size = actions[self.get_action_index("hidden_size")]
        num_hops = actions[self.get_action_index("num_hops")]
        normalization = actions[self.get_action_index("normalization")]
        adj_norm = actions[self.get_action_index("adj_normalization")]

        if format == "one":
            attention_type_att=actions[self.get_action_index("attention_type_att")]
            # mlp_layers = actions[6]
            self.model = SSRGCN(
                num_features=args.num_features,
                hidden_size=hidden_size,
                num_classes=args.num_classes,
                attention=attention_type,
                attention_att=attention_type_att,
                activation=activation,
                normalization=normalization,
                nhop=num_hops,
                dropout=args.dropout,
                node_dropout=args.node_dropout,
                alpha=args.alpha,
                nhead=num_heads,
                adj_normalization=adj_norm,
            )

        else:
            self.model = SRGCN(
                num_features=args.num_features,
                hidden_size=hidden_size,
                attention=attention_type,
                activation=activation,
                normalization=normalization,
                nhop=num_hops,
                dropout=args.dropout,
                node_dropout=args.node_dropout,
                alpha=args.alpha,
                nhead=num_heads,
                adj_normalization=adj_norm,
            )
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.model(x, edge_index, edge_attr)


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, args):
        super(MLP, self).__init__()
        self.dropout = args.dropout
        self.W = nn.Linear(in_feats, out_feats)

    def forward(self, x, *args, **kwargs):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W(x)
        x = F.relu(x)
        return F.log_softmax(x, dim=-1)