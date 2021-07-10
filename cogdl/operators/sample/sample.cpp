    #include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor indptr, torch::Tensor indices, torch::Tensor node_idx, int64_t num_neighbors, bool replace) {
    auto indptr_data = indptr.data_ptr<int64_t>();
    auto indices_data = indices.data_ptr<int64_t>();
    auto node_idx_data = node_idx.data_ptr<int64_t>();

    auto out_indptr = torch::empty(node_idx.numel() + 1, indptr.options());
    auto out_indptr_data = out_indptr.data_ptr<int64_t>();
    out_indptr_data[0] = 0;

    auto num_input_nodes = node_idx.numel();

    std::vector<int64_t> new_indices;
    std::vector<int64_t> new_edges;
    std::vector<int64_t> new_node_idx;

    auto assoc = torch::full({indptr.size(0) - 1}, -1, node_idx.options());
    assoc.index_copy_(0, node_idx, torch::arange(num_input_nodes, node_idx.options()));
    auto assoc_data = assoc.data_ptr<int64_t>();

//    std::unordered_map<int64_t, int64_t> node_id_map;

    int64_t node = 0;
    for(int64_t i = 0; i < node_idx.numel(); i++) {
        node = node_idx_data[i];
//        node_id_map[node] = i;
        new_node_idx.push_back(node);
    }


    int64_t row_start, row_count, edge, src_node;
    int64_t num_nodes = node_idx.numel();
    if(num_neighbors < 0){
        for(int64_t i = 0; i < num_input_nodes; i++) {
            node = node_idx_data[i];
            row_start = indptr_data[node];
            row_count = indptr_data[node+1] - row_start;

            for(int k = 0; k < row_count; k++) {
                edge = row_start + k;
                src_node = indices_data[edge];

                if(assoc_data[src_node] == -1) {
                    assoc_data[src_node] = num_nodes;
                    new_node_idx.push_back(src_node);
                    num_nodes++;
                }
                new_indices.push_back(assoc_data[src_node]);

//                if(node_id_map.count(src_node) == 0) {
//                    node_id_map[src_node] = num_nodes;
//                    new_node_idx.push_back(src_node);
//                    num_nodes++;
//                }
//                new_indices.push_back(node_id_map[src_node]);

                new_edges.push_back(edge);
            }
            out_indptr_data[i+1] = row_count + out_indptr_data[i];
        }
    } else if(replace) {
        for(int64_t i = 0; i < num_input_nodes; i++) {
            node = node_idx_data[i];
            row_start = indptr_data[node];
            row_count = indptr_data[node+1] - row_start;

            for(int64_t k = 0; k < num_neighbors; k++) {
                edge = row_start + rand() % row_count;
                src_node = indices_data[edge];

                if(assoc_data[src_node] == -1) {
                    assoc_data[src_node] = num_nodes;
                    new_node_idx.push_back(src_node);
                    num_nodes++;
                }
                new_indices.push_back(assoc_data[src_node]);

//                if(node_id_map.count(src_node) == 0) {
//                    node_id_map[src_node] = num_nodes;
//                    new_node_idx.push_back(src_node);
//                    num_nodes++;
//                }
//                new_indices.push_back(node_id_map[src_node]);

                new_edges.push_back(edge);
            }
            out_indptr_data[i+1] = num_neighbors + out_indptr_data[i];
        }
    } else {
        for(int64_t i = 0; i < num_input_nodes; i++) {
            node = node_idx_data[i];
            row_start = indptr_data[node];
            row_count = indptr_data[node+1] - row_start;

            std::unordered_set<int64_t> perm;
            if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                perm.insert(j);
            } else { // See: https://www.nowherenearithaca.com/2013/05/
                    //      robert-floyds-tiny-and-beautiful.html
            for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
            }

            for(const int64_t &p : perm) {
                edge = row_start + p;
                src_node = indices_data[edge];

                if(assoc_data[src_node] == -1) {
                    assoc_data[src_node] = num_nodes;
                    new_node_idx.push_back(src_node);
                    num_nodes++;
                }
                new_indices.push_back(assoc_data[src_node]);

//                if(node_id_map.count(src_node) == 0) {
//                    node_id_map[src_node] = num_nodes;
//                    new_node_idx.push_back(src_node);
//                    num_nodes++;
//                }
//                new_indices.push_back(node_id_map[src_node]);

                new_edges.push_back(edge);
            }
            out_indptr_data[i+1] = perm.size() + out_indptr_data[i];
        }
    }

    int64_t num_edges = out_indptr_data[num_input_nodes];
    int64_t out_num_nodes = new_node_idx.size();
    
    auto out_nodes = torch::from_blob(new_node_idx.data(), {out_num_nodes}, indices.options()).clone();
    auto out_indices = torch::from_blob(new_indices.data(), {num_edges}, indices.options()).clone();
    auto out_edges = torch::from_blob(new_edges.data(), {num_edges}, indices.options()).clone();

    return std::make_tuple(out_indptr, out_indices, out_nodes, out_edges); 
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_cpu(torch::Tensor indptr, torch::Tensor indices, torch::Tensor node_idx) {
    auto node_idx_data = node_idx.data_ptr<int64_t>();
    auto indptr_data = indptr.data_ptr<int64_t>();
    auto indices_data = indices.data_ptr<int64_t>();

    int64_t num_nodes = node_idx.numel();

    auto out_indptr = torch::empty(num_nodes + 1, indptr.options());
    auto out_indptr_data = out_indptr.data_ptr<int64_t>();
    out_indptr_data[0] = 0;

    std::vector<int64_t> new_indices;
    std::vector<int64_t> new_edges;

    auto assoc = torch::full({indptr.size(0) - 1}, -1, node_idx.options());
    assoc.index_copy_(0, node_idx, torch::arange(num_nodes, node_idx.options()));
    auto assoc_data = assoc.data_ptr<int64_t>();

    int64_t node, src, src_new;
    int64_t row_start, row_end, num_edges = 0;
    for(int64_t i = 0; i < num_nodes; i++) {
        node = node_idx_data[i];
        row_start = indptr_data[node];
        row_end = indptr_data[node+1];
        
        for(int64_t k = row_start; k < row_end; k++) {
            src = indices_data[k];
            src_new = assoc_data[src];
            if(src_new > -1) {
                new_indices.push_back(src_new);
                new_edges.push_back(k);
                num_edges++;
            } 
        }
        out_indptr_data[i+1] = num_edges;
    }
    auto out_indices = torch::from_blob(new_indices.data(), {num_edges}, indptr.options()).clone();
    auto out_edges = torch::from_blob(new_edges.data(), {num_edges}, indptr.options()).clone();
    auto out_nodes = torch::arange(0, num_nodes);

    return std::make_tuple(out_indptr, out_indices, out_nodes, out_edges);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
coo2csr_cpu(torch::Tensor row, torch::Tensor col, torch::Tensor val, int64_t num_nodes) {
    auto nnz = row.size(0);
    auto row_ptr = torch::zeros(num_nodes+1, row.options());
    auto col_ind = torch::empty(nnz, col.options());
    auto out_val = torch::empty(nnz, val.options());

    auto row_ptr_data = row_ptr.data_ptr<int64_t>();
    auto col_ind_data = col_ind.data_ptr<int64_t>();
    auto out_val_data = out_val.data_ptr<float>();
    auto row_data = row.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
    auto val_data = val.data_ptr<float>();

    for(int64_t i = 0; i < nnz; i++) {
        row_ptr_data[row_data[i]]++;
    }

    int64_t temp, row_num, dest;
    for(int64_t i = 0, cumsum = 0; i < num_nodes; i++) {
        temp = row_ptr_data[i];
        row_ptr_data[i] = cumsum;
        cumsum += temp;
    }
    row_ptr_data[num_nodes] = nnz;
    for(int64_t i = 0; i < nnz; i++) {
        row_num = row_data[i];
        dest = row_ptr_data[row_num];
        col_ind_data[dest] = col_data[i];
        out_val_data[dest] = val_data[i];

        row_ptr_data[row_num]++;
    }

    for(int64_t i = 0, last = 0; i <= num_nodes; i++){
        temp = row_ptr_data[i];
        row_ptr_data[i] = last;
        last = temp;
    }
    return std::make_tuple(row_ptr, col_ind, out_val);
}


std::tuple<torch::Tensor, torch::Tensor>
coo2csr_cpu_index(torch::Tensor row, torch::Tensor col, int64_t num_nodes) {
    auto nnz = row.size(0);
    auto row_ptr = torch::zeros(num_nodes+1, row.options());
    auto col_ind = torch::empty(nnz, col.options());

    auto row_ptr_data = row_ptr.data_ptr<int64_t>();
    auto col_ind_data = col_ind.data_ptr<int64_t>();
    auto row_data = row.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();

    for(int64_t i = 0; i < nnz; i++) {
        row_ptr_data[row_data[i]]++;
    }

    int64_t temp, row_num, dest;
    for(int64_t i = 0, cumsum = 0; i < num_nodes; i++) {
        temp = row_ptr_data[i];
        row_ptr_data[i] = cumsum;
        cumsum += temp;
    }
    row_ptr_data[num_nodes] = nnz;
    for(int64_t i = 0; i < nnz; i++) {
        row_num = row_data[i];
        dest = row_ptr_data[row_num];
        col_ind_data[dest] = i;

        row_ptr_data[row_num]++;
    }

    for(int64_t i = 0, last = 0; i <= num_nodes; i++){
        temp = row_ptr_data[i];
        row_ptr_data[i] = last;
        last = temp;
    }
    return std::make_tuple(row_ptr, col_ind);
}


PYBIND11_MODULE(sampler, m) {
    m.def("sample_adj", &sample_adj, "sample neighborhood");
    m.def("subgraph", &subgraph_cpu, "subgraph");
    m.def("coo2csr_cpu", &coo2csr_cpu, "coo2csr");
    m.def("coo2csr_cpu_index", &coo2csr_cpu_index, "coo2csr with index");
}
