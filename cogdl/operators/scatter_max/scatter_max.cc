#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void assertTensor(torch::Tensor &T, c10::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

std::vector<torch::Tensor> scatter_max_fp_cuda(torch::Tensor rowptr,
                                               torch::Tensor colind,
                                               torch::Tensor node_feature);

torch::Tensor scatter_max_bp_cuda(torch::Tensor node_feature,
                                  torch::Tensor max_mask, long num_nodes);

std::vector<torch::Tensor> scatter_max(torch::Tensor rowptr,
                                       torch::Tensor colind,
                                       torch::Tensor node_feature) {
  assertTensor(rowptr, torch::kInt32);
  assertTensor(colind, torch::kInt32);
  assertTensor(node_feature, torch::kFloat32);
  return scatter_max_fp_cuda(rowptr, colind, node_feature);
}

torch::Tensor scatter_max_bp(torch::Tensor node_feature,
                             torch::Tensor max_mask) {
  assertTensor(node_feature, torch::kFloat32);
  assertTensor(max_mask, torch::kInt32);
  return scatter_max_bp_cuda(node_feature, max_mask);
}

PYBIND11_MODULE(scatter_max, m) {
  m.doc() = "scatter max kernel";
  m.def("scatter_max_fp", &scatter_max, "scatter max forward");
  m.def("scatter_max_bp", &scatter_max_bp, "scatter max backward");
}
