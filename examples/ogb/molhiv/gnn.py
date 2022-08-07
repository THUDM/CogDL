import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from cogdl.models import BaseModel
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

import math

### node encoder and edge encoder
class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
                node_feat           [0, 1]
                node_is_attributed  [2]
                node_dfs_order      [3]
                node_depth          [4]

        Output:
            emb_dim-dimensional vector

    '''
    def __init__(self, emb_dim, num_nodetypes=100, num_nodeattributes=10100, max_depth=20):
        super(ASTNodeEncoder, self).__init__()
        self.max_depth = max_depth
        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x):
        depth = x[:, 4].clone()
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:,0]) + self.attribute_encoder(x[:,1]) + self.depth_encoder(depth)

def get_node_encoder(emb_dim, dataset_name):
    if dataset_name in ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-pcqm4m', 'ogbg-pcqm4mv2']:
        return AtomEncoder(emb_dim = emb_dim)
    elif dataset_name == 'ogbg-ppa':
        return torch.nn.Embedding(1, emb_dim)
    elif dataset_name == 'ogbg-code2':
        return ASTNodeEncoder(emb_dim = emb_dim)

def get_edge_encoder(emb_dim, dataset_name):
    if dataset_name in ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-pcqm4m', 'ogbg-pcqm4mv2']:
        return BondEncoder(emb_dim = emb_dim)
    elif dataset_name == 'ogbg-ppa':
        return torch.nn.Linear(7, emb_dim)
    elif dataset_name == 'ogbg-code2':
        return torch.nn.Linear(2, emb_dim)

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, dataset_name = None):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = get_edge_encoder(emb_dim, dataset_name)
        

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, dataset_name = None):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = get_edge_encoder(emb_dim, dataset_name)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', dataset_name = None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.dataset_name = dataset_name

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = get_node_encoder(emb_dim, dataset_name)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, dataset_name))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, dataset_name))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_index = torch.cat([edge_index[0].view(1, -1), edge_index[1].view(1, -1)], dim=0)

        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', dataset_name = None):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = get_node_encoder(emb_dim, dataset_name)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, dataset_name))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, dataset_name))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim),
                    torch.nn.BatchNorm1d(2*emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU()
                    )
                )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_index = torch.cat([edge_index[0].view(1, -1), edge_index[1].view(1, -1)], dim=0)



        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GNN(BaseModel):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", dataset_name = None, num_vocab = 5000, max_seq_len = 5):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.dataset_name = dataset_name

        # only for code2
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, dataset_name = dataset_name)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, dataset_name = dataset_name)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
        
        if dataset_name != 'ogbg-code2':
            if graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if graph_pooling == "set2set":
                for i in range(max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_tasks))
            else:
                for i in range(max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_tasks))

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        if self.dataset_name == 'ogbg-code2':
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](h_graph))
            output = torch.cat(pred_list, dim=1).view(h_graph.shape[0] * self.max_seq_len, -1)
        else:
            output = self.graph_pred_linear(h_graph)
        
        if self.dataset_name == 'ogbg-pcqm4m' or self.dataset_name == 'ogbg-pcqm4mv2':
            if self.training:
                return output
            else:
                # At inference time, we clamp the value between 0 and 20
                return torch.clamp(output, min=0, max=20)
        else:
            return output


