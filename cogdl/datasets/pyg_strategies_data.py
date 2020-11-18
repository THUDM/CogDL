"""
    This file is borrowed from https://github.com/snap-stanford/pretrain-gnns/
"""
from cogdl.datasets import register_dataset
import random
import zipfile
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from cogdl.data import download_url
import os.path as osp
from itertools import repeat, product, chain

# ================
# Dataset utils
# ================

def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

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
        edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                        attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                        attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
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
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)

    try:
        species_id = int(nx_node_ids[0].split('.')[0])  # nx node id is of the form:
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)    # dummy species
        # id is 0

    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                        dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                        dtype=torch.long)
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
            G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
                       w6=w6, w7=w7)
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
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

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
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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

        edge_set = set([str(data.edge_index[0,i].cpu().item()) + "," + str(data.edge_index[1,i].cpu().item()) for i in range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2,5*num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5*num_edges):
            node1 = redandunt_sample[0,i].cpu().item()
            node2 = redandunt_sample[1,i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges/2:
                break

        data.negative_edge_index = redandunt_sample[:,sampled_ind]
        
        return data


class MaskEdge:
    """Borrowed from https://github.com/snap-stanford/pretrain-gnns/"""
    def __init__(self, mask_rate):
        """
        Assume edge_attr is of the form:
        [w1, w2, w3, w4, w5, w6, w7, self_loop, mask]
        :param mask_rate: % of edges to be masked
        """
        self.mask_rate = mask_rate

    def __call__(self, data, masked_edge_indices=None):
        if masked_edge_indices == None:
            # sample x distinct edges to be masked, based on mask rate. But
            # will sample at least 1 edge
            num_edges = int(data.edge_index.size()[1] / 2)  # num unique edges
            sample_size = int(num_edges * self.mask_rate + 1)
            # during sampling, we only pick the 1st direction of a particular
            # edge pair
            masked_edge_indices = [2 * i for i in random.sample(range(
                num_edges), sample_size)]

        data.masked_edge_idx = torch.tensor(np.array(masked_edge_indices))

        # create ground truth edge features for the edges that correspond to
        # the masked indices
        mask_edge_labels_list = []
        for idx in masked_edge_indices:
            mask_edge_labels_list.append(data.edge_attr[idx].view(1, -1))
        data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

        # created new masked edge_attr, where both directions of the masked
        # edges have masked edge type. For message passing in gcn

        # append the 2nd direction of the masked edges
        all_masked_edge_indices = masked_edge_indices + [i + 1 for i in
                                                         masked_edge_indices]
        for idx in all_masked_edge_indices:
            data.edge_attr[idx] = torch.tensor(np.array([0, 0, 0, 0, 0,
                                                             0, 0, 0, 1]),
                                                      dtype=torch.float)

        return data

class MaskAtom:
    """Borrowed from https://github.com/snap-stanford/pretrain-gnns/"""
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

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

        if root_idx == None:
            if self.center == True:
                root_idx = data.center_node_idx.item()
            else:
                root_idx = random.sample(range(num_atoms), 1)[0]

        # in the PPI case, the subgraph is the entire PPI graph
        data.x_substruct = data.x
        data.edge_attr_substruct = data.edge_attr
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = data.center_node_idx


        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_atoms)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)   # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

    def __repr__(self):
        return '{}(l1={}, center={})'.format(self.__class__.__name__,
                                              self.l1, self.center)

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
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) == 0:
            l2_node_idxes = range(num_atoms)
            context_node_idxes = set(l1_node_idxes).symmetric_difference(
                set(l2_node_idxes))

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
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) <= 0:
            context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

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
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


# ==================
# DataLoader utils
# ==================


class BatchFinetune(Data):
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
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'center_node_idx']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchMasking(Data):
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
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key  == 'masked_edge_idx':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchAE(Data):
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
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]
        for key in keys:
            batch[key] = []

        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
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
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderFinetune(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderFinetune, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchFinetune.from_data_list(data_list),
            **kwargs)


class DataLoaderMasking(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


class DataLoaderAE(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)


class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)


# ==========
# Dataset
# ==========


@register_dataset("test_bio")
class TestBioDataset(InMemoryDataset):
    def __init__(self,
                 data_type="unsupervised",
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(TestBioDataset, self).__init__(root, transform, pre_transform, pre_filter)
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

        edge_attr = torch.zeros(num_edges * num_graphs, 9)
        for idx, val in enumerate(torch.randint(0, 9, size=(num_edges * num_graphs,))):
            edge_attr[idx][val] = 1.
        self.data = Data(
            x=torch.ones(num_graphs * num_nodes, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        self.data.center_node_idx = torch.randint(0, num_nodes, size=(num_graphs,))
        self.slices = {
            "x": torch.arange(0, (num_graphs + 1) * num_nodes, num_nodes),
            "edge_index": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
            "edge_attr": torch.arange(0, (num_graphs + 1) * num_edges, num_edges),
            "center_node_idx": torch.arange(num_graphs+1),
        }

        if data_type == "supervised":
            pretrain_tasks = 10
            downstream_tasks = 5
            go_target_pretrain = torch.zeros(pretrain_tasks * num_graphs)
            go_target_downstream = torch.zeros(downstream_tasks * num_graphs)
            go_target_pretrain[torch.arange(0, pretrain_tasks*num_graphs, pretrain_tasks)] = 1
            go_target_downstream[torch.arange(0, downstream_tasks*num_graphs, downstream_tasks)] = 1
            self.data.go_target_downstream = go_target_downstream
            self.data.go_target_pretrain = go_target_pretrain
            self.slices["go_target_pretrain"] = torch.arange(0, (num_graphs + 1) * pretrain_tasks)
            self.slices["go_target_downstream"] = torch.arange(0, (num_graphs + 1) * downstream_tasks)

@register_dataset("test_chem")
class TestChemDataset(InMemoryDataset):
    def __init__(self,
                 data_type="unsupervised",
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
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

        self.data = Data(
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
        
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


@register_dataset("bio")
class BioDataset(InMemoryDataset):
    def __init__(self,
                 data_type="unsupervised",
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.data_type = data_type
        self.url = "https://cloud.tsinghua.edu.cn/f/c865b1d61348489e86ac/?dl=1"
        self.root = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", "BIO")
        super(BioDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        if not empty:
            if data_type == "unsupervised":
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['processed.zip']

    @property
    def processed_file_names(self):
        return ['supervised_data_processed.pt', 'unsupervised_data_processed.pt']

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]),'r')
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))

@register_dataset("chem")
class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 data_type="unsupervised",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        self.data_type = data_type
        self.url = "https://cloud.tsinghua.edu.cn/f/2cac04ee904e4b54b4b2/?dl=1"
        self.root = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", "CHEM")

        super(MoleculeDataset, self).__init__(self.root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            if data_type == "unsupervised":
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        return ['processed.zip']

    @property
    def processed_file_names(self):
        return ['supervised_data_processed.pt', 'unsupervised_data_processed.pt']

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]),'r')
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))

# ==========
# Dataset for finetuning
# ==========

@register_dataset("bace")
class BACEDataset(InMemoryDataset):
    def __init__(self,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        self.url = "https://cloud.tsinghua.edu.cn/f/253270b278f4465380f1/?dl=1"
        self.root = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", "BACE")

        super(BACEDataset, self).__init__(self.root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        return ['processed.zip']

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]),'r')
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))

@register_dataset("bbbp")
class BBBPDataset(InMemoryDataset):
    def __init__(self,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        self.url = "https://cloud.tsinghua.edu.cn/f/ab8ff4d0a68c40a38956/?dl=1"
        self.root = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", "BBBP")

        super(BBBPDataset, self).__init__(self.root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        return ['processed.zip']

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        download_url(self.url, self.raw_dir, name="processed.zip")

    def process(self):
        zfile = zipfile.ZipFile(osp.join(self.raw_dir, self.raw_file_names[0]),'r')
        for filename in zfile.namelist():
            print("unzip file: " + filename)
            zfile.extract(filename, osp.join(self.processed_dir))
