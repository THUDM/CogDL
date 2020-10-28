import math
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.parsing.preprocessing import *
from texttable import Texttable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from tqdm import tqdm

"""
    utils.py
"""


def args_print(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1.0 / (r[0] + 1) if r.size else 0.0 for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def randint():
    return np.random.randint(2 ** 32 - 1)


def feature_OAG(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]

        if "node_emb" in graph.node_feature[_type]:
            feature[_type] = np.array(
                list(graph.node_feature[_type].loc[idxs, "node_emb"]), dtype=np.float
            )
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate(
            (
                feature[_type],
                list(graph.node_feature[_type].loc[idxs, "emb"]),
                np.log10(
                    np.array(
                        list(graph.node_feature[_type].loc[idxs, "citation"])
                    ).reshape(-1, 1)
                    + 0.01
                ),
            ),
            axis=1,
        )

        times[_type] = tims
        indxs[_type] = idxs

        if _type == "paper":
            attr = np.array(
                list(graph.node_feature[_type].loc[idxs, "title"]), dtype=np.str
            )
    return feature, times, indxs, attr


def feature_reddit(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]

        feature[_type] = np.array(
            list(graph.node_feature[_type].loc[idxs, "emb"]), dtype=np.float
        )
        times[_type] = tims
        indxs[_type] = idxs

        if _type == "def":
            attr = feature[_type]
    return feature, times, indxs, attr


def load_gnn(_dict):
    out_dict = {}
    for key in _dict:
        if "gnn" in key:
            out_dict[key[4:]] = _dict[key]
    return OrderedDict(out_dict)


"""
    data.py
"""


def defaultDictDict():
    return {}


def defaultDictList():
    return []


def defaultDictInt():
    return defaultdict(int)


def defaultDictDictInt():
    return defaultdict(defaultDictInt)


def defaultDictDictDictInt():
    return defaultdict(defaultDictDictInt)


def defaultDictDictDictDictInt():
    return defaultdict(defaultDictDictDictInt)


def defaultDictDictDictDictDictInt():
    return defaultdict(defaultDictDictDictDictInt)


class Graph:
    def __init__(self):
        super(Graph, self).__init__()
        """
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame

            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        """
        self.node_forward = defaultdict(defaultDictDict)
        self.node_bacward = defaultdict(defaultDictList)
        self.node_feature = defaultdict(defaultDictList)

        """
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        """
        # self.edge_list = defaultdict(  # target_type
        #     lambda: defaultdict(  # source_type
        #         lambda: defaultdict(  # relation_type
        #             lambda: defaultdict(  # target_id
        #                 lambda: defaultdict(int)  # source_id(  # time
        #             )
        #         )
        #     )
        # )
        self.edge_list = defaultDictDictDictDictDictInt()
        self.times = {}

    def add_node(self, node):
        nfl = self.node_forward[node["type"]]
        if node["id"] not in nfl:
            self.node_bacward[node["type"]] += [node]
            ser = len(nfl)
            nfl[node["id"]] = ser
            return ser
        return nfl[node["id"]]

    def add_edge(
        self, source_node, target_node, time=None, relation_type=None, directed=True
    ):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        """
            Add bi-directional edges with different relation type
        """
        self.edge_list[target_node["type"]][source_node["type"]][relation_type][
            edge[1]
        ][edge[0]] = time
        if directed:
            self.edge_list[source_node["type"]][target_node["type"]][
                "rev_" + relation_type
            ][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node["type"]][target_node["type"]][relation_type][
                edge[0]
            ][edge[1]] = time
        self.times[time] = True

    def update_node(self, node):
        nbl = self.node_bacward[node["type"]]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas

    def get_types(self):
        return list(self.node_feature.keys())


def sample_subgraph(
    graph,
    time_range,
    sampled_depth=2,
    sampled_number=8,
    inp=None,
    feature_extractor=feature_OAG,
):
    """
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    """
    layer_data = defaultdict(lambda: {})  # target_type  # {target_id: [ser, time]}
    budget = defaultdict(  # source_type
        lambda: defaultdict(lambda: [0.0, 0])  # source_id  # [sampled_score, time]
    )
    new_layer_adj = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(lambda: [])  # relation_type  # [target_id, source_id]
        )
    )
    """
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    """

    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == "self" or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(
                        list(adl.keys()), sampled_number, replace=False
                    )
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if (
                        source_time > np.max(list(time_range.keys()))
                        or source_id in layer_data[source_type]
                    ):
                        continue
                    budget[source_type][source_id][0] += 1.0 / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

    """
        First adding the sampled nodes then updating budget.
    """
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    """
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    """
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                """
                    Directly sample all the nodes
                """
                sampled_ids = np.arange(len(keys))
            else:
                """
                    Sample based on accumulated degree
                """
                score = np.array(list(budget[source_type].values()))[:, 0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(
                    len(score), sampled_number, p=score, replace=False
                )
            sampled_keys = keys[sampled_ids]
            """
                First adding the sampled nodes then updating budget.
            """
            for k in sampled_keys:
                layer_data[source_type][k] = [
                    len(layer_data[source_type]),
                    budget[source_type][k][1],
                ]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)
    """
        Prepare feature, time and adjacency matrix for the sampled graph
    """
    feature, times, indxs, texts = feature_extractor(layer_data, graph)

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(lambda: [])  # relation_type  # [target_id, source_id]
        )
    )
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]["self"] += [[_ser, _ser]]
    """
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    """
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        """
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        """
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [
                                [target_ser, source_ser]
                            ]
    return feature, times, edge_list, indxs, texts


def to_torch(feature, time, edge_list, graph):
    """
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    """
    node_dict = {}
    node_feature = []
    node_type = []
    node_time = []
    edge_index = []
    edge_type = []
    edge_time = []

    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    if "fake_paper" in feature:
        node_dict["fake_paper"] = [node_num, node_dict["paper"][1]]
        node_num += len(feature["fake_paper"])
        types += ["fake_paper"]

    for t in types:
        node_feature += list(feature[t])
        node_time += list(time[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]

    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict["self"] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(
                    edge_list[target_type][source_type][relation_type]
                ):
                    tid, sid = (
                        ti + node_dict[target_type][0],
                        si + node_dict[source_type][0],
                    )
                    edge_index += [[sid, tid]]
                    edge_type += [edge_dict[relation_type]]
                    """
                        Our time ranges from 1900 - 2020, largest span is 120.
                    """
                    edge_time += [node_time[tid] - node_time[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type = torch.LongTensor(node_type)
    edge_time = torch.LongTensor(edge_time)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    return (
        node_feature,
        node_type,
        edge_time,
        edge_index,
        edge_type,
        node_dict,
        edge_dict,
    )


"""
    conv.py
"""


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout=0.2,
        use_norm=True,
        use_RTE=True,
        **kwargs
    ):
        super(HGTConv, self).__init__(aggr="add", **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dim = 0
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        """
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        """
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_time=edge_time,
        )

    def message(
        self,
        edge_index_i,
        node_inp_i,
        node_inp_j,
        node_type_i,
        node_type_j,
        edge_type,
        edge_time,
    ):
        """
            j: source, i: target; <j, i>
        """
        data_size = edge_index_i.size(0)
        """
            Create Attention and Message tensor beforehand.
        """
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = node_type_j == int(source_type)
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    """
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    """
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    """
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    """
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = self.emb(node_inp_j[idx], edge_time[idx])

                    """
                        Step 1: Heterogeneous Mutual Attention
                    """
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(
                        k_mat.transpose(1, 0), self.relation_att[relation_type]
                    ).transpose(1, 0)
                    res_att[idx] = (
                        (q_mat * k_mat).sum(dim=-1)
                        * self.relation_pri[relation_type]
                        / self.sqrt_dk
                    )
                    """
                        Step 2: Heterogeneous Message Passing
                    """
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(
                        v_mat.transpose(1, 0), self.relation_msg[relation_type]
                    ).transpose(1, 0)
        """
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        """
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        """
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        """
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.a_linears[target_type](aggr_out[idx])
            """
                Add skip connection with learnable weight self.skip[t_id]
            """
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](
                    trans_out * alpha + node_inp[idx] * (1 - alpha)
                )
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return self.drop(res)

    def __repr__(self):
        return "{}(in_dim={}, out_dim={}, num_types={}, num_types={})".format(
            self.__class__.__name__,
            self.in_dim,
            self.out_dim,
            self.num_types,
            self.num_relations,
        )


class RelTemporalEncoding(nn.Module):
    """
        Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0.0, n_hid * 2, 2.0)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid
        )
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid
        )
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t)))


class GeneralConv(nn.Module):
    def __init__(
        self,
        conv_name,
        in_hid,
        out_hid,
        num_types,
        num_relations,
        n_heads,
        dropout,
        use_norm=True,
        use_RTE=True,
    ):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == "hgt":
            self.base_conv = HGTConv(
                in_hid,
                out_hid,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm,
                use_RTE,
            )
        elif self.conv_name == "gcn":
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == "gat":
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == "hgt":
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == "gcn":
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == "gat":
            return self.base_conv(meta_xs, edge_index)


"""
    model.py
"""


class GNN(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hid,
        num_types,
        num_relations,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="hgt",
        prev_norm=False,
        last_norm=False,
        use_RTE=True,
    ):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(
                    conv_name,
                    n_hid,
                    n_hid,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm=prev_norm,
                    use_RTE=use_RTE,
                )
            )
        self.gcs.append(
            GeneralConv(
                conv_name,
                n_hid,
                n_hid,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm=last_norm,
                use_RTE=use_RTE,
            )
        )

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class GPT_GNN(nn.Module):
    def __init__(
        self,
        gnn,
        rem_edge_list,
        attr_decoder,
        types,
        neg_samp_num,
        device,
        neg_queue_size=0,
    ):
        super(GPT_GNN, self).__init__()
        self.types = types
        self.gnn = gnn
        self.params = nn.ModuleList()
        self.neg_queue_size = neg_queue_size
        self.link_dec_dict = {}
        self.neg_queue = {}
        for source_type in rem_edge_list:
            self.link_dec_dict[source_type] = {}
            self.neg_queue[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                print(source_type, relation_type)
                matcher = Matcher(gnn.n_hid, gnn.n_hid)
                self.neg_queue[source_type][relation_type] = torch.FloatTensor([]).to(
                    device
                )
                self.link_dec_dict[source_type][relation_type] = matcher
                self.params.append(matcher)
        self.attr_decoder = attr_decoder
        self.init_emb = nn.Parameter(torch.randn(gnn.in_dim))
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.neg_samp_num = neg_samp_num

    def neg_sample(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = []
        keys = {key: True for key in pos_node_list}
        tot = 0
        for node_id in souce_node_list:
            if node_id not in keys:
                neg_nodes += [node_id]
                tot += 1
            if tot == self.neg_samp_num:
                break
        return neg_nodes

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)

    def link_loss(
        self,
        node_emb,
        rem_edge_list,
        ori_edge_list,
        node_dict,
        target_type,
        use_queue=True,
        update_queue=False,
    ):
        losses = 0
        ress = []
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <= 8:
                    continue
                ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]

                target_ids, positive_source_ids = (
                    rem_edges[:, 0].reshape(-1, 1),
                    rem_edges[:, 1].reshape(-1, 1),
                )
                n_nodes = len(target_ids)
                source_node_ids = np.unique(ori_edges[:, 1])

                negative_source_ids = [
                    self.neg_sample(
                        source_node_ids,
                        ori_edges[ori_edges[:, 0] == t_id][:, 1].tolist(),
                    )
                    for t_id in target_ids
                ]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])

                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                source_ids = torch.LongTensor(
                    np.concatenate((positive_source_ids, negative_source_ids), axis=-1)
                    + node_dict[source_type][0]
                )
                emb = node_emb[source_ids]

                if (
                    use_queue
                    and len(self.neg_queue[source_type][relation_type]) // n_nodes > 0
                ):
                    tmp = self.neg_queue[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)

                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                target_emb = node_emb[target_ids.reshape(-1)]
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:, 0].mean()
                if (
                    update_queue
                    and "L1" not in relation_type
                    and "L2" not in relation_type
                ):
                    tmp = self.neg_queue[source_type][relation_type]
                    self.neg_queue[source_type][relation_type] = torch.cat(
                        [node_emb[source_node_ids].detach(), tmp], dim=0
                    )[: int(self.neg_queue_size * n_nodes)]
        return -losses / len(ress), ress

    def text_loss(self, reps, texts, w2v_model, device):
        def parse_text(texts, w2v_model, device):
            idxs = []
            pad = w2v_model.wv.vocab["eos"].index
            for text in texts:
                idx = []
                for word in ["bos"] + preprocess_string(text) + ["eos"]:
                    if word in w2v_model.wv.vocab:
                        idx += [w2v_model.wv.vocab[word].index]
                idxs += [idx]
            mxl = np.max([len(s) for s in idxs]) + 1
            inp_idxs = []
            out_idxs = []
            masks = []
            for i, idx in enumerate(idxs):
                inp_idxs += [idx + [pad for _ in range(mxl - len(idx) - 1)]]
                out_idxs += [idx[1:] + [pad for _ in range(mxl - len(idx))]]
                masks += [
                    [1 for _ in range(len(idx))]
                    + [0 for _ in range(mxl - len(idx) - 1)]
                ]
            return (
                torch.LongTensor(inp_idxs).transpose(0, 1).to(device),
                torch.LongTensor(out_idxs).transpose(0, 1).to(device),
                torch.BoolTensor(masks).transpose(0, 1).to(device),
            )

        inp_idxs, out_idxs, masks = parse_text(texts, w2v_model, device)
        pred_prob = self.attr_decoder(inp_idxs, reps.repeat(inp_idxs.shape[0], 1, 1))
        return self.ce(pred_prob[masks], out_idxs[masks]).mean()

    def feat_loss(self, reps, out):
        return -self.attr_decoder(reps, out).mean()


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return "{}(n_hid={}, n_out={})".format(
            self.__class__.__name__, self.n_hid, self.n_out
        )


class Matcher(nn.Module):
    """
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    """

    def __init__(self, n_hid, n_out, temperature=0.1):
        super(Matcher, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.sqrt_hd = math.sqrt(n_out)
        self.drop = nn.Dropout(0.2)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.cache = None
        self.temperature = temperature

    def forward(self, x, ty, use_norm=True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd

    def __repr__(self):
        return "{}(n_hid={})".format(self.__class__.__name__, self.n_hid)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, n_word, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(nhid, nhid, nlayers)
        self.encoder = nn.Embedding(n_word, nhid)
        self.decoder = nn.Linear(nhid, n_word)
        self.adp = nn.Linear(ninp + nhid, nhid)

    def forward(self, inp, hidden=None):
        emb = self.encoder(inp)
        if hidden is not None:
            emb = torch.cat((emb, hidden), dim=-1)
            emb = F.gelu(self.adp(emb))
        output, _ = self.rnn(emb)
        decoded = self.decoder(self.drop(output))
        return decoded

    def from_w2v(self, w2v):
        initrange = 0.1
        self.encoder.weight.data = w2v
        self.decoder.weight = self.encoder.weight

        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False


"""
    preprocess_reddit.py
"""


def preprocess_dataset(dataset) -> Graph:
    graph_reddit = Graph()
    el = defaultdict(
        lambda: defaultdict(lambda: int)
    )  # target_id  # source_id(  # time
    for i, j in tqdm(dataset.data.edge_index.t()):
        el[i.item()][j.item()] = 1

    target_type = "def"
    graph_reddit.edge_list["def"]["def"]["def"] = el
    n = list(el.keys())
    degree = np.zeros(np.max(n) + 1)
    for i in n:
        degree[i] = len(el[i])
    x = np.concatenate((dataset.data.x.numpy(), np.log(degree).reshape(-1, 1)), axis=-1)
    graph_reddit.node_feature["def"] = pd.DataFrame({"emb": list(x)})

    idx = np.arange(len(graph_reddit.node_feature[target_type]))
    # np.random.seed(43)
    np.random.shuffle(idx)

    print(dataset.data.x.shape)

    graph_reddit.pre_target_nodes = idx[: int(len(idx) * 0.7)]
    graph_reddit.train_target_nodes = idx
    graph_reddit.valid_target_nodes = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
    graph_reddit.test_target_nodes = idx[int(len(idx) * 0.9) :]
    # graph_reddit.pre_target_nodes = []
    # graph_reddit.train_target_nodes = []
    # graph_reddit.valid_target_nodes = []
    # graph_reddit.test_target_nodes = []
    # for i in range(len(graph_reddit.node_feature[target_type])):
    #     if dataset.data.train_mask[i]:
    #         graph_reddit.pre_target_nodes.append(i)
    #         graph_reddit.train_target_nodes.append(i)
    #     if dataset.data.val_mask[i]:
    #         graph_reddit.valid_target_nodes.append(i)
    #     if dataset.data.test_mask[i]:
    #         graph_reddit.test_target_nodes.append(i)
    #
    # graph_reddit.pre_target_nodes = np.array(graph_reddit.pre_target_nodes)
    # graph_reddit.train_target_nodes = np.array(graph_reddit.train_target_nodes)
    # graph_reddit.valid_target_nodes = np.array(graph_reddit.valid_target_nodes)
    # graph_reddit.test_target_nodes = np.array(graph_reddit.test_target_nodes)
    graph_reddit.train_mask = dataset.data.train_mask
    graph_reddit.val_mask = dataset.data.val_mask
    graph_reddit.test_mask = dataset.data.test_mask

    # print(np.sum(dataset.data.train_mask.numpy()))
    # print(np.sum(dataset.data.val_mask.numpy()))
    # print(np.sum(dataset.data.test_mask.numpy()))
    # print(graph_reddit.train_target_nodes.shape)

    graph_reddit.y = dataset.data.y
    return graph_reddit
