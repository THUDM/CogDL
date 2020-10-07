import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch

from cogdl.data import Data

class Sampler:
    def __init__(self, data, args_params):
        self.data = data
        self.num_nodes = self.data.x.size()[0]
        self.num_edges = self.data.edge_index.size()[1]

    def sample(self):
        pass

class SAINTSampler(Sampler):
    def __init__(self, data, args_params):
        super().__init__(data, args_params)
        self.adj = sp.coo_matrix((np.ones(self.num_edges), (self.data.edge_index[0], self.data.edge_index[1])), 
                            shape = (self.num_nodes, self.num_nodes)).tocsr()
        self.node_train = np.arange(1, self.num_nodes + 1) * self.data.train_mask.numpy()
        self.node_train = self.node_train[self.node_train != 0] - 1

        self.sample_coverage = args_params["sample_coverage"]
        self.estimate()

    def estimate(self):
        # estimation of loss / aggregation normalization factors
        # -------------------------------------------------------------
        # For some special sampler, no need to estimate norm factors, we can calculate
        # the node / edge probabilities directly.
        # However, for integrity of the framework, we follow the same procedure
        # for all samplers:
        #   1. sample enough number of subgraphs
        #   2. update the counter for each node / edge in the training graph
        #   3. estimate norm factor alpha and lambda
        self.subgraphs_indptr = []
        self.subgraphs_indices = []
        self.subgraphs_data = []
        self.subgraphs_nodes = []
        self.subgraphs_edge_index = []

        self.norm_loss_train = np.zeros(self.num_nodes)
        self.norm_aggr_train = np.zeros(self.num_edges)
        self.norm_loss_test = np.ones(self.num_nodes) / self.num_nodes
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))

        tot_sampled_nodes = 0
        while True:
            tot_sampled_nodes += self.gen_subgraph()
            print("\rGenerating subgraphs %.2lf%%" % (tot_sampled_nodes * 100 / self.data.num_nodes / self.sample_coverage), end = "", flush = True)
            if tot_sampled_nodes > self.sample_coverage * self.num_nodes:
                break
        num_subg = len(self.subgraphs_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_nodes[i]] += 1
        for v in range(self.data.num_nodes):
            i_s = self.adj.indptr[v]
            i_e = self.adj.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s : i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

    def gen_subgraph(self):
        _indptr, _indices, _data, _v, _edge_index = self.sample()
        self.subgraphs_indptr.append(_indptr)
        self.subgraphs_indices.append(_indices)
        self.subgraphs_data.append(_data)
        self.subgraphs_nodes.append(_v)
        self.subgraphs_edge_index.append(_edge_index)
        return len(_v)

    def sample(self):
        pass

    def extract_subgraph(self, edge_idx, directed = True):
        edge_idx = np.unique(edge_idx)
        subg_edge = self.data.edge_index.t()[edge_idx].numpy()
        if not directed:
            subg_edge = np.concatenate((subg_edge, subg_edge[:,[1, 0]]))
        subg_edge = np.unique(subg_edge, axis = 0)
        #get nodes whose degree != 0
        nodes = np.zeros(self.num_nodes, dtype = int)
        for e in subg_edge:
            nodes[e[0]] = 1
            nodes[e[1]] = 1
        if not directed:
            edge_idx = []
            for e in subg_edge:
                for i in range(self.adj.indptr[e[0]], self.adj.indptr[e[0] + 1]):
                    if self.adj.indices[i] == e[1]:
                        edge_idx.append(i)
            edge_idx = np.array(edge_idx, dtype = int)
        node_idx = np.arange(1, self.num_nodes + 1) * np.array(nodes)
        node_idx = node_idx[node_idx != 0] - 1

        #mapping nodes to new indices
        orig2subg = {n: i for i, n in enumerate(node_idx)}
        indptr = np.zeros(node_idx.size + 1, dtype = int)
        for ind, u in enumerate(subg_edge[:, 0]):
            if ind + 1 > indptr[orig2subg[u] + 1]:
                indptr[orig2subg[u] + 1] = ind + 1
        for i in range(1, node_idx.size + 1):
            if indptr[i] == 0:
                indptr[i] = indptr[i - 1]
        indices = subg_edge[:, 1]
        for i in range(len(indices)):
            indices[i] = orig2subg[indices[i]]
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge.size // 2
        return indptr, indices, data, node_idx, edge_idx

    def get_subgraph(self, phase, require_norm = True):
        """
        Generate one minibatch for model. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'valid' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            mode                str, can be 'train', 'valid', 'test'
            require_norm        boolean

        Outputs:
            data                Data object, modeling the sampled subgraph
            data.norm_aggr      aggregation normalization
            data.norm_loss      normalization normalization
        """
        if phase in ['val', 'test']:
            node_subgraph = np.arange(self.data.num_nodes)
            data = self.data
            if require_norm:
                data.norm_aggr = torch.ones(self.num_edges)
                data.norm_loss = self.norm_loss_test
        else:
            if len(self.subgraphs_nodes) == 0:
                self.gen_subgraph()

            node_subgraph = self.subgraphs_nodes.pop()
            edge_subgraph = self.subgraphs_edge_index.pop()
            num_nodes_subgraph = node_subgraph.size
            adj = sp.csr_matrix((self.subgraphs_data.pop(), self.subgraphs_indices.pop(), self.subgraphs_indptr.pop()), 
                            shape =(num_nodes_subgraph, num_nodes_subgraph))

            if require_norm:
                adj.data[:] = self.norm_aggr_train[edge_subgraph][:]
                #normalization
                D = adj.sum(1).flatten()
                norm_diag = sp.dia_matrix((1 / D, 0), shape = adj.shape)
                adj = norm_diag.dot(adj)
                adj.sort_indices()

            adj = adj.tocoo()
            data = Data(self.data.x[node_subgraph], 
                        torch.LongTensor(np.vstack((adj.row, adj.col))),
                        None if self.data.edge_attr is None else self.data.edge_attr[edge_subgraph], 
                        self.data.y[node_subgraph], 
                        None if self.data.pos is None else self.data.pos[node_subgraph])

            if require_norm:
                data.norm_aggr = torch.FloatTensor(adj.data)
                data.norm_loss = self.norm_loss_train[node_subgraph]
            data.train_mask = self.data.train_mask[node_subgraph]
            data.val_mask = self.data.val_mask[node_subgraph]
            data.test_mask = self.data.test_mask[node_subgraph]

        return data

class NodeSampler(SAINTSampler):
    def __init__(self, data, args_params):
        self.node_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)
        
    def sample(self):
        node_idx = np.random.choice(np.arange(self.num_nodes), self.node_num_subgraph)
        node_idx = np.unique(node_idx)
        node_idx.sort()
        orig2subg = {n: i for i, n in enumerate(node_idx)}
        indptr = np.zeros(node_idx.size + 1)
        indices = []
        subg_edge_index = []
        for nid in node_idx:
            idx_s, idx_e = self.adj.indptr[nid], self.adj.indptr[nid + 1]
            neighs = self.adj.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        return indptr, indices, data, node_idx, subg_edge_index
        
class EdgeSampler(SAINTSampler):
    def __init__(self, data, args_params):
        self.edge_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)

    def sample(self):
        edge_idx = np.random.choice(np.arange(self.num_edges), self.edge_num_subgraph)
        return self.extract_subgraph(edge_idx)

class RWSampler(SAINTSampler):
    def __init__(self, data, args_params):
        self.num_walks = args_params["num_walks"]
        self.walk_length = args_params["walk_length"]
        super().__init__(data, args_params)

    def sample(self):
        edge_idx = []
        for walk in range(self.num_walks):
            u = np.random.randint(self.num_nodes)
            for step in range(self.walk_length):
                idx_s = self.adj.indptr[u]
                idx_e = self.adj.indptr[u + 1]
                e = np.random.randint(idx_s, idx_e)
                edge_idx.append(e)
                u = self.adj.indices[e]
        
        return self.extract_subgraph(np.array(edge_idx))

class MRWSampler(SAINTSampler):
    def __init__(self, data, args_params):
        self.size_frontier = args_params["size_frontier"]
        self.edge_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)

    def sample(self):
        frontier = np.random.choice(np.arange(self.num_nodes), self.size_frontier)
        deg = self.adj.indptr[frontier + 1] - self.adj.indptr[frontier]
        deg_sum = np.sum(deg)
        edge_idx = []
        for i in range(self.edge_num_subgraph):
            val = np.random.randint(deg_sum)
            id = 0
            while val >= deg[id]:
                val -= deg[id]
                id += 1
            nid = frontier[id]
            idx_s, idx_e = self.adj.indptr[nid], self.adj.indptr[nid + 1]
            e = np.random.randint(idx_s, idx_e)
            edge_idx.append(e)
            v = self.adj.indices[e]
            frontier[id] = v
            deg_sum -= deg[id]
            deg[id] = self.adj.indptr[v + 1] - self.adj.indptr[v]
            deg_sum += deg[id]

        return self.extract_subgraph(np.array(edge_idx))

class LayerSampler(Sampler):
    def __init__(self, data, model, params_args):
        super().__init__(data, params_args)
        self.model = model
        self.sample_one_layer = self.model._sample_one_layer
        self.sample = self.model.sampling

    def get_batches(self, train_nodes, train_labels, batch_size=64, shuffle=True):
        if shuffle:
            random.shuffle(train_nodes)
        total = train_nodes.shape[0]
        for i in range(0, total, batch_size):
            if i + batch_size <= total:
                cur_nodes = train_nodes[i: i+batch_size]
                cur_labels = train_labels[cur_nodes]
                yield cur_nodes, cur_labels

"""class FastGCNSampler(LayerSampler):
    def __init__(self, data, params_args):
        super().__init__(data, params_args)

    def generate_adj(self, sample1, sample2):
        edgelist = []
        mapping = {}
        for i in range(len(sample1)):
            mapping[sample1[i]] = i

        for i in range(len(sample2)):
            nodes = self.adj[sample2[i]]
            for node in nodes:
                if node in mapping:
                    edgelist.append([mapping[node], i])
        edgetensor = torch.LongTensor(edgelist)
        valuetensor = torch.ones(edgetensor.shape[0]).float()
        t = torch.sparse_coo_tensor(
            edgetensor.t(), valuetensor, (len(sample1), len(sample2))
        )
        return t

    def sample_one_layer(self, sampled, sample_size):
        total = []
        for node in sampled:
            total.extend(self.adj[node])
        total = list(set(total))
        if sample_size < len(total):
            total = random.sample(total, sample_size)
        return total

    def sample(self, x, v, num_layers):
        all_support = [[] for _ in range(num_layers)]
        sampled = v.detach().cpu().numpy()
        for i in range(num_layers - 1, -1, -1):
            cur_sampled = self.sample_one_layer(sampled, self.sample_size[i])
            all_support[i] = self.generate_adj(sampled, cur_sampled).to(x.device)
            sampled = cur_sampled

        return x[torch.LongTensor(sampled).to(x.device)], all_support, 0

class ASGCNSampler(LayerSampler):
    def __init__(self, data, params_args):
        super().__init__(data, params_args)

    def set_w(w_s0, w_s1):
        self.w_s0 = w_s0
        self.w_s1 = w_s1

    def set_adj(self, edge_index, num_nodes):
        self.sparse_adj = sparse.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        self.num_nodes = num_nodes
        self.adj = self.compute_adjlist(self.sparse_adj)
        self.adj = torch.tensor(self.adj)

    def compute_adjlist(self, sp_adj, max_degree=32):
        num_data = sp_adj.shape[0]
        adj = num_data + np.zeros((num_data+1, max_degree), dtype=np.int32)

        for v in range(num_data):
            neighbors = np.nonzero(sp_adj[v, :])[1]
            len_neighbors = len(neighbors)
            if len_neighbors > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
                adj[v] = neighbors
            else:
                adj[v, :len_neighbors] = neighbors

        return adj

    def from_adjlist(self, adj):
        u_sampled, index = torch.unique(torch.flatten(adj), return_inverse=True)

        row = (torch.range(0, index.shape[0]-1) / adj.shape[1]).long().to(adj.device)
        col = index
        values = torch.ones(index.shape[0]).float().to(adj.device)
        indices = torch.cat([row.unsqueeze(1), col.unsqueeze(1)], axis=1).t()
        dense_shape = (adj.shape[0], u_sampled.shape[0])

        support = torch.sparse_coo_tensor(indices, values, dense_shape)

        return support, u_sampled.long()

    def _sample_one_layer(self, x, adj, v, sample_size):
        support, u = self.from_adjlist(adj)


        h_v = torch.sum(torch.matmul(x[v], self.w_s1))
        h_u = torch.matmul(x[u], self.w_s0)
        attention = (F.relu(h_v + h_u) + 1) * (1.0 / sample_size)
        g_u = F.relu(h_u) + 1

        p1 = attention * g_u
        p1 = p1.cpu()

        if self.num_nodes in u:
            p1[u == self.num_nodes] = 0
        p1 = p1 / torch.sum(p1)

        samples = torch.multinomial(p1, sample_size, False)
        u_sampled = u[samples]
            
        support_sampled = torch.index_select(support, 1, samples)

        return u_sampled, support_sampled
    
    def sample(self, x, v, num_layers):
        all_support = [[] for _ in range(num_layers)]
        sampled = v
        x = torch.cat((x, torch.zeros(1, x.shape[1]).to(x.device)), dim=0)
        for i in range(num_layers - 1, -1, -1):
            cur_sampled, cur_support = self.sample_one_layer(x, self.adj[sampled], sampled, self.sample_size[i])
            all_support[i] = cur_support.to(x.device)
            sampled = cur_sampled

        return x[sampled.to(x.device)], all_support, 0"""
