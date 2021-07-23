"""
    This file is borrowed from https://github.com/snap-stanford/pretrain-gnns/
"""
from cogdl.datasets import register_dataset
import random
import zipfile
import networkx as nx
import numpy as np

import torch
from cogdl.utils import download_url
import os.path as osp

from cogdl.data import Graph, MultiGraphDataset, Adjacency

# ================
# Dataset utils
# ================


def nx_to_graph_data_obj(
    g, center_id, allowable_features_downstream=None, allowable_features_pretrain=None, node_id_to_go_labels=None
):
    n_nodes = g.number_of_nodes()

    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # in a particular ordering. Will be used as a mapping to convert
    # between nx node ids and data obj node indices

    x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # we don't have any node labels, so set to dummy 1. dim n_nodes x 1

    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [
            attr_dict["w1"],
            attr_dict["w2"],
            attr_dict["w3"],
            attr_dict["w4"],
            attr_dict["w5"],
            attr_dict["w6"],
            attr_dict["w7"],
            0,
            0,
        ]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)

    try:
        species_id = int(nx_node_ids[0].split(".")[0])  # nx node id is of the form:
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except Exception:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)  # dummy species
        # id is 0

    # construct data obj
    data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.species_id = species_id
    data.center_node_idx = center_node_idx

    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature), dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature), dtype=torch.long)
    return data


def graph_data_obj_to_nx(data):
    G = nx.Graph()

    # edges
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        w1, w2, w3, w4, w5, w6, w7, _, _ = edge_attr[j].astype(bool)
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6, w7=w7)
    return G


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx, bond_dir_idx=bond_dir_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    # num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node["atom_num_idx"], node["chirality_tag_idx"]]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["bond_type_idx"], edge["bond_dir_idx"]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class NegativeEdge:
    """Borrowed from https://github.com/snap-stanford/pretrain-gnns/"""

    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set(
            [
                str(data.edge_index[0, i].cpu().item()) + "," + str(data.edge_index[1, i].cpu().item())
                for i in range(data.edge_index.shape[1])
            ]
        )

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if edge_str not in edge_set and edge_str not in sampled_edge_set and node1 != node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class ExtractSubstructureContextPair:
    def __init__(self, l1, center=True):
        self.center = center
        self.l1 = l1

        if self.l1 == 0:
            self.l1 = -1

    def __call__(self, data, root_idx=None):
        num_atoms = data.x.size()[0]
        G = graph_data_obj_to_nx(data)

        if root_idx is None:
            if self.center is True:
                root_idx = data.center_node_idx.item()
            else:
                root_idx = random.sample(range(num_atoms), 1)[0]

        # in the PPI case, the subgraph is the entire PPI graph
        data.x_substruct = data.x
        data.edge_attr_substruct = data.edge_attr
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = data.center_node_idx

        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_atoms)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)  # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes
            ]
            data.overlap_context_substruct_idx = torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

    def __repr__(self):
        return "{}(l1={}, center={})".format(self.__class__.__name__, self.l1, self.center)


class ChemExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2
        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """
        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx is None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) == 0:
            l2_node_idxes = range(num_atoms)
            context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))

        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) <= 0:
            context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes
            ]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return "{}(k={},l1={}, l2={})".format(self.__class__.__name__, self.k, self.l1, self.l2)


# ==================
# DataLoader utils
# ==================


def build_batch(batch, data_list, num_nodes_cum, num_edges_cum, keys):
    for key in batch.keys:
        item = batch[key][0]
        if torch.is_tensor(item):
            # batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, item))
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        elif isinstance(item, Adjacency):
            target = Adjacency()
            for k in item.keys:
                if k == "row" or k == "col":
                    _item = torch.cat(
                        [x[k] + num_nodes_cum[i] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None)
                    )
                elif k == "row_ptr":
                    _item = torch.cat(
                        [x[k][:-1] + num_edges_cum[i] for i, x in enumerate(batch[key][:-1])],
                        dim=item.cat_dim(k, None),
                    )
                    _item = torch.cat([_item, batch[key][-1][k] + num_edges_cum[-2]], dim=item.cat_dim(k, None))
                else:
                    _item = torch.cat([x[k] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None))
                target[k] = _item
            batch[key] = target.to(item.device)
    return batch


class BatchMasking(Graph):
    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0
        num_nodes_cum = [0]
        num_edges_cum = [0]

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                elif key == "masked_edge_idx":
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index[0].shape[0]
            num_nodes_cum.append(num_nodes)
            num_edges_cum.append(data.edge_index[0].shape[0])

        # for key in keys:
        #     batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch = build_batch(batch, data_list, num_nodes_cum, num_edges_cum, keys)
        # batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "face", "masked_atom_indices", "connected_edge_indices"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchAE(Graph):
    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ["edge_index", "negative_edge_index"]:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        assert "batch" not in keys
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Graph):
    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        batch = BatchSubstructContext()
        keys = [
            "center_substruct_idx",
            "edge_attr_substruct",
            "edge_index_substruct",
            "x_substruct",
            "overlap_context_substruct_idx",
            "edge_attr_context",
            "edge_index_context",
            "x_context",
        ]
        for key in keys:
            batch[key] = []

        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                # batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long)
                )
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                # batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        # batch = build_batch(batch, data_list, num_nodes_cum, num_edges_cum)
        # batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in [
            "edge_index",
            "edge_index_substruct",
            "edge_index_context",
            "overlap_context_substruct_idx",
            "center_substruct_idx",
        ]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderAE(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset, batch_size, shuffle, collate_fn=lambda data_list: BatchAE.from_data_list(data_list), **kwargs
        )


class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs,
        )


# ==========
# Dataset
# ==========


@register_dataset("test_bio")
class TestBioDataset(MultiGraphDataset):
    def __init__(self, data_type="unsupervised", root="testbio", transform=None, pre_transform=None, pre_filter=None):
        super(TestBioDataset, self).__init__(root, transform, pre_transform, pre_filter)
        num_nodes = 20
        num_edges = 40
        num_graphs = 200

        def cycle_index(num, shift):
            arr = torch.arange(num) + shift
            arr[-shift:] = torch.arange(shift)
            return arr

        upp = torch.cat([torch.cat((torch.arange(0, num_nodes), torch.arange(0, num_nodes)))] * num_graphs)
        dwn = torch.cat([torch.cat((torch.arange(0, num_nodes), cycle_index(num_nodes, 1)))] * num_graphs)
        edge_index = torch.stack([upp, dwn])

        edge_attr = torch.zeros(num_edges * num_graphs, 9)
        for idx, val in enumerate(torch.randint(0, 9, size=(num_edges * num_graphs,))):
            edge_attr[idx][val] = 1.0
        self.data = Graph(
            x=torch.ones(num_graphs * num_nodes, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        self.data.center_node_idx = torch.randint(0, num_nodes, size=(num_graphs,))

        self.slices = {
            "x": torch.arange(0, (num_graphs + 1) * num_nodes, num_nodes),
            "edge_index": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
            "edge_attr": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
            "center_node_idx": torch.arange(num_graphs + 1),
        }

        if data_type == "supervised":
            pretrain_tasks = 10
            downstream_tasks = 5
            # go_target_pretrain = torch.zeros(pretrain_tasks * num_graphs)
            # go_target_downstream = torch.zeros(downstream_tasks * num_graphs)

            go_target_downstream = torch.randint(0, 2, (downstream_tasks * num_graphs,))
            go_target_pretrain = torch.randint(0, 2, (pretrain_tasks * num_graphs,))

            # go_target_pretrain[torch.randint(0, pretrain_tasks*num_graphs, pretrain_tasks)] = 1
            # go_target_downstream[torch.arange(0, downstream_tasks*num_graphs, downstream_tasks)] = 1
            self.data.go_target_downstream = go_target_downstream
            self.data.go_target_pretrain = go_target_pretrain
            self.slices["go_target_pretrain"] = torch.arange(0, (num_graphs + 1) * pretrain_tasks)
            self.slices["go_target_downstream"] = torch.arange(0, (num_graphs + 1) * downstream_tasks)

    def _download(self):
        pass

    def _process(self):
        pass


@register_dataset("test_chem")
class TestChemDataset(MultiGraphDataset):
    def __init__(self, data_type="unsupervised", root="testchem", transform=None, pre_transform=None, pre_filter=None):
        super(TestChemDataset, self).__init__(root, transform, pre_transform, pre_filter)
        num_nodes = 10
        num_edges = 10
        num_graphs = 100

        def cycle_index(num, shift):
            arr = torch.arange(num) + shift
            arr[-shift:] = torch.arange(shift)
            return arr

        upp = torch.cat([torch.arange(0, num_nodes)] * num_graphs)
        dwn = torch.cat([cycle_index(num_nodes, 1)] * num_graphs)
        edge_index = torch.stack([upp, dwn])

        edge_attr = torch.zeros(num_edges * num_graphs, 2)
        x = torch.zeros(num_graphs * num_nodes, 2)
        for idx, val in enumerate(torch.randint(0, 6, size=(num_edges * num_graphs,))):
            edge_attr[idx][0] = val
        for idx, val in enumerate(torch.randint(0, 3, size=(num_edges * num_graphs,))):
            edge_attr[idx][1] = val
        for idx, val in enumerate(torch.randint(0, 120, size=(num_edges * num_graphs,))):
            x[idx][0] = val
        for idx, val in enumerate(torch.randint(0, 3, size=(num_edges * num_graphs,))):
            x[idx][1] = val

        self.data = Graph(
            x=x.to(torch.long),
            edge_index=edge_index.to(torch.long),
            edge_attr=edge_attr.to(torch.long),
        )

        self.slices = {
            "x": torch.arange(0, (num_graphs + 1) * num_nodes, num_nodes),
            "edge_index": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
            "edge_attr": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
        }

        if data_type == "supervised":
            pretrain_tasks = 10
            go_target_pretrain = torch.zeros(pretrain_tasks * num_graphs) - 1
            for i in range(num_graphs):
                val = np.random.randint(0, pretrain_tasks)
                go_target_pretrain[i * pretrain_tasks + val] = 1
            self.data.y = go_target_pretrain
            self.slices["y"] = torch.arange(0, (num_graphs + 1) * pretrain_tasks, pretrain_tasks)

    def _download(self):
        pass

    def _process(self):
        pass


@register_dataset("bio")
class BioDataset(MultiGraphDataset):
    def __init__(
        self,
        data_type="unsupervised",
        empty=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        data_path="data",
    ):
        self.data_type = data_type
        self.url = "https://cloud.tsinghua.edu.cn/f/c865b1d61348489e86ac/?dl=1"
        self.root = osp.join(data_path, "BIO")
        super(BioDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        if not empty:
            if data_type == "unsupervised":
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["processed.zip"]

    @property
    def processed_file_names(self):
        return ["supervised_data_processed.pt", "unsupervised_data_processed.pt"]

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]), "r")
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))


@register_dataset("chem")
class MoleculeDataset(MultiGraphDataset):
    def __init__(
        self,
        data_type="unsupervised",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        data_path="data",
    ):
        self.data_type = data_type
        self.url = "https://cloud.tsinghua.edu.cn/f/2cac04ee904e4b54b4b2/?dl=1"
        self.root = osp.join(data_path, "CHEM")

        super(MoleculeDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            if data_type == "unsupervised":
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["processed.zip"]

    @property
    def processed_file_names(self):
        return ["supervised_data_processed.pt", "unsupervised_data_processed.pt"]

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]), "r")
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))


# ==========
# Dataset for finetuning
# ==========


def convert(data):
    if not hasattr(data, "_adj"):
        g = Graph()
        for key in data.keys:
            if "adj" in key:
                g["_" + key] = data[key]
            else:
                g[key] = data[key]
        return g
    else:
        return data


@register_dataset("bace")
class BACEDataset(MultiGraphDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None, empty=False, data_path="data"):
        self.url = "https://cloud.tsinghua.edu.cn/d/c6bd3405569b4fab9c4a/files/?p=%2Fprocessed.zip&dl=1"
        self.root = osp.join(data_path, "BACE")

        super(BACEDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data = convert(self.data)

    @property
    def raw_file_names(self):
        return ["processed.zip"]

    @property
    def processed_file_names(self):
        return ["processed.pt"]

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]), "r")
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))


@register_dataset("bbbp")
class BBBPDataset(MultiGraphDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None, empty=False, data_path="data"):
        self.url = "https://cloud.tsinghua.edu.cn/d/9db9e16a949b4877bb4e/files/?p=%2Fprocessed.zip&dl=1"
        self.root = osp.join(data_path, "BBBP")

        super(BBBPDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data = convert(self.data)

    @property
    def raw_file_names(self):
        return ["processed.zip"]

    @property
    def processed_file_names(self):
        return ["processed.pt"]

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]), "r")
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))
